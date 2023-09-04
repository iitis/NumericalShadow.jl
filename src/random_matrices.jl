export random_pure, random_overlap
using CUDA

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

function random_pure(::Type{T}, d, batchsize) where {T}
    ψd = CUDA.randn(T, d, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims = 1)))
    ψd = ψd .* norm_invs
    return ψd
end

function random_overlap(::Type{T}, d, batchsize, q::Real) where {T}
    x = random_pure(T, d, batchsize)
    y = random_pure(T, d, batchsize)
    xp = gram_schmidt_step(x, y)
    return sqrt(1 - q^2) * xp + q * x, x
end

function random_unitary(d, batchsize)

    function cat_kernel(C, vs)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        i > length(vs) && return
        C[:, :, i] = vs[i]
        return
    end

    function outer_prod_kernel(C, A, B)
        #this probably can be impreoved using shared memory
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        mx, my, _ = size(C)
        idx > prod(size(C)) && return
        i = mod1(idx, mx)
        idx = cld(idx, mx)
        j = mod1(idx, my)
        idx = cld(idx, my)
        k = idx
        C[i, j, k] = A[i, k] * B[j, k]
        return
    end

    g = CUDA.randn(d, d, batchsize)
    q, r = CUBLAS.geqrf_batched!([view(g, :, :, i) for i=1:batchsize])
    rr = r[1]
    rr = tril(rr)
    rr[diagind(rr)] = 1
    pp = proj.(eachcol(rr))
    qq = prod([CuArray(I, 4, 4) - pp[i] * q[1][i] for i=1:4]) # this is the q for the first matrix in batch
    proper_r = triu(r[1])
end

function random_max_ent(d, batchsize)
    x = CUDA.randn(ComplexF32, d, d, batchsize)
    tau, A = CUDA.CUBLAS.geqrf_batched(collect(eachslice(x, dims = 3)))
    q, r = reconstruct_qr(tau, A)
    return qr_fix!(q, r)
end
