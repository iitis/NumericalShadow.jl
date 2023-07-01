using Base.Threads
import MLUtils: unsqueeze
using CUDA
import NNlibCUDA: ⊠, batched_adjoint

CUDA.allowscalar(false)


function random_pure(::Type{T}, d, batchsize) where {T}
    ψd = CUDA.randn(T, d, 1, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims=1)))
    ψd = ψd ⊠ norm_invs;
    return ψd
end

function random_unitary(d, batchsize)
end

function random_max_ent(d, batchsize)
    x = CUDA.randn(ComplexF32, d, d, batchsize)
    tau, A = CUDA.CUBLAS.geqrf_batched(collect(eachslice(x, dims=3)));
    q, r = reconstruct_qr(tau, A)
    return qr_fix!(q, r)
end
