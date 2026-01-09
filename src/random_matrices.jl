export random_pure, random_overlap, random_unitary, random_max_ent
using KernelAbstractions, LinearAlgebra
using TensorCast

function gram_schmidt_step(backend, x, y)
   d = sum(conj.(x) .* y, dims=1)
   y .- d .* x 
end

function random_pure(backend, ::Type{T}, d, batchsize) where {T}
    _random_pure(backend, T, d, batchsize)
end

function _random_pure(::CPU, ::Type{T}, d, batchsize) where {T}
    ψd = randn(T, d, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims = 1)))
    ψd .*= norm_invs
    return ψd
end

function random_overlap(backend, ::Type{T}, d, batchsize, q::Real) where {T}
    x = random_pure(backend, T, d, batchsize)
    y = random_pure(backend, T, d, batchsize)
    xp = gram_schmidt_step(backend, x, y)
    norm_xp = sqrt.(sum(real.(xp .* conj.(xp)), dims=1))
    xp ./= norm_xp
    return sqrt(1 - q^2) * xp + q * x, x
end


function random_unitary(backend, ::Type{T}, d, batchsize) where {T}
    _random_unitary(backend, T, d, batchsize)
end

function _random_unitary(::CPU, ::Type{T}, d, batchsize) where {T}
    # Generic fallback using loops
    Q = zeros(T, d, d, batchsize)
    Threads.@threads for i in 1:batchsize
        g = randn(T, d, d)
        q_val, r_val = qr(g)
        d_diag = sign.(diag(r_val))
        d_diag[d_diag .== 0] .= 1
        d_mat = Diagonal(d_diag)
        Q[:, :, i] .= Matrix(q_val) * d_mat
    end
    return Q
end

function random_max_ent(backend, ::Type{T}, d, batchsize) where {T}
    Q = random_unitary(backend, T, d, batchsize)
    ψ = reduce(hcat, vec.(eachslice(Q; dims=3))) ./ T(d) # Check normalization factor?
    return ψ
end
