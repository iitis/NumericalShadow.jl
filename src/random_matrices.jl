export random_pure, random_overlap, random_unitary, random_max_ent, random_product
using CUDA, LinearAlgebra

CUDA.allowscalar(false)

function gram_schmidt_step(x, y)
    d, batchsize = size(x)
    overlaps = CUDA.zeros(eltype(x), 1, batchsize)
    # <y, x> x
    conj!(x)
    CUDA.CUBLAS.gemv_strided_batched!('N', 1.0, reshape(y, 1, d, batchsize), x, 1, overlaps)
    conj!(x)
    proj = overlaps .* x
    return y - proj
end

function random_pure(::Type{T}, d::Int, batchsize::Int) where {T}
    ψd = CUDA.randn(T, d, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims = 1)))
    ψd = ψd .* norm_invs
    return ψd
end

function random_product(::Type{T}, d::Int, batchsize::Int) where {T}
    local_d = isqrt(d)
    ψd = random_pure(T, local_d, batchsize)
    ϕd = random_pure(T, local_d, batchsize)
    ξd = CUDA.zeros(T, d, batchsize)
    my_kron!(ξd, ψd, ϕd)
    return ξd
end

function random_overlap(::Type{T}, d, batchsize, q::Real) where {T}
    x = random_pure(T, d, batchsize)
    y = random_pure(T, d, batchsize)
    xp = gram_schmidt_step(x, y)
    return sqrt(1 - q^2) * xp + q * x, x
end

# function random_unitary(::Type{T}, d, batchsize) where {T}
#     g = CUDA.randn(T, d, d, batchsize)
#     _tau, _r = CUBLAS.geqrf_batched!([view(g, :, :, i) for i=1:batchsize])
#     r = CUDA.zeros(T, d, d, batchsize)
#     tau = CUDA.zeros(T, d, batchsize)
#     correction_factors = CUDA.zeros(T, d, batchsize)
#     CUDA.@time @views Threads.@threads for i=1:batchsize
#         correction_factors[:, i] = sign.(diag(_r[i]))
#         _r[i][diagind(_r[i])] .= 1
#         r[:, :, i] = tril(_r[i])
#         tau[:, i] = _tau[i]
#     end
#     id = CuArray(I, d, d)
#     @cast b[k, l, i, m] := id - r[k, i, m] * conj(r[l, i, m]) * tau[i, m]
#     Q = CUDA.zeros(T, d, d, batchsize)
#     CUDA.@time @views Threads.@threads for i=1:batchsize
#         Q[:, :, i] = b[:, :, 1, i] * b[:, :, 2, i]
#         for j=3:size(b,  3)
#             Q[:, :, i] *= b[:, :, j, i]
#         end
#     end
#     @cast Q[i, j, k] = Q[i, j, k] * correction_factors[i, k]
#     return Q
# end

function random_max_ent(::Type{T}, d, batchsize) where {T}
    Q = random_unitary(T, d, batchsize)
    ψ = reduce(hcat, vec.(eachslice(Q; dims=3))) ./ d
    return ψ
end
