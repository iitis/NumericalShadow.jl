module NumericalShadowCUDAExt

using CUDA
using KernelAbstractions
using NumericalShadow
using LinearAlgebra
using TensorCast

import NumericalShadow: _random_pure, _random_unitary, move_to_backend, gram_schmidt_step

function move_to_backend(::CUDABackend, data)
    CuArray(data)
end

function gram_schmidt_step(::CUDABackend, x, y)
    d, batchsize = size(x)
    overlaps = CUDA.zeros(eltype(x), 1, batchsize)
    # <y, x> x
    conj!(x)
    CUDA.CUBLAS.gemv_strided_batched!('N', 1.0, reshape(y, 1, d, batchsize), x, 1, overlaps)
    conj!(x)
    proj = overlaps .* x
    return y - proj
end

function _random_pure(::CUDABackend, ::Type{T}, d, batchsize) where {T}
    ψd = CUDA.randn(T, d, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims = 1)))
    ψd .*= norm_invs
    return ψd
end

@kernel inbounds=true function reconstruct_Q_unitary!(Q, A, tau_mat, phases, d, batchsize)
    batch_idx = @index(Global, Linear)

    if batch_idx <= batchsize 
        @inbounds for row in 1:d
            @inbounds for col in 1:d
                Q[row, col, batch_idx] = row == col ? one(eltype(Q)) : zero(eltype(Q))
            end
        end

        @inbounds for k in d:-1:1
             τk = tau_mat[k, batch_idx]

            for col in 1:d
                dot = zero(eltype(Q))
                for r in k:d
                    v_r = r == k ? one(eltype(Q)) : A[r, k, batch_idx]
                    dot += conj(v_r) * Q[r, col, batch_idx]
                end
                for r in k:d
                    v_r = r == k ? one(eltype(Q)) : A[r, k, batch_idx]
                    Q[r, col, batch_idx] -= τk * v_r * dot
                end
            end

            phases[k, batch_idx] = A[k, k, batch_idx] / sqrt(abs2(A[k, k, batch_idx]))

        end
    end
end

@kernel function apply_phases!(Q, phases, d, batchsize)
    batch_idx = @index(Global, Linear)

    if batch_idx <= batchsize
        @inbounds for row in 1:d
            @inbounds for col in 1:d
                Q[row, col, batch_idx] *= phases[row, batch_idx]
            end
        end
    end
end

function _random_unitary(backend::CUDABackend, ::Type{T}, d, batchsize) where {T}
    Z = CUDA.randn(T, d, d, batchsize)
    # Z is modified in-place by geqrf_batched! to store vectors v, 
    # but the diagonal contains R diagonal elements (not 1s of v).
    # The Householder vectors v have 1 on diagonal (implicit).
    tau, _ = CUDA.CUBLAS.geqrf_batched!([view(Z, :, :, i) for i=1:batchsize])

    Q = CUDA.zeros(T, d, d, batchsize)

    # We need to copy tau to a matrix to use it in the kernel easily without weird indexing
    tau_mat = CUDA.zeros(T, d, batchsize)
    phases = CUDA.zeros(T, d, batchsize)

    # Copy tau vector of CuArrays into a single CuMatrix
    # This loop is on CPU, orchestrating data movement. 
    # tau is a Vector{CuArray}.
    for i in 1:batchsize
        tau_mat[:, i] .= tau[i]
    end

    kernel = reconstruct_Q_unitary!(backend)
    kernel(Q, Z, tau_mat, phases, d, batchsize, ndrange=batchsize)
        
    KernelAbstractions.synchronize(backend)
    apply_phases!(backend)(Q, phases, d, batchsize, ndrange=batchsize)
    KernelAbstractions.synchronize(backend)

    return Q
end

end
