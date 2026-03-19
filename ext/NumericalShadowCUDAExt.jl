module NumericalShadowCUDAExt

using CUDA
using KernelAbstractions
using NumericalShadow
using LinearAlgebra

import NumericalShadow:
    _random_pure,
    _random_unitary,
    _reconstruct_q_from_compact_qr!,
    gram_schmidt_step,
    move_to_backend

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

    _reconstruct_q_from_compact_qr!(backend, Q, Z, tau_mat, phases, d, batchsize)
    return Q
end

end
