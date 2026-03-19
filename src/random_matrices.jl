using KernelAbstractions, LinearAlgebra

function _validate_positive_int(name::Symbol, value)
    if !(value isa Integer) || value <= 0
        throw(ArgumentError("`$name` must be a positive integer, got $value"))
    end
    return Int(value)
end

function _validate_q(q::Real)
    if !(0 <= q <= 1)
        throw(ArgumentError("`q` must satisfy 0 <= q <= 1, got $q"))
    end
    return q
end

function gram_schmidt_step(backend, x, y)
    overlap = sum(conj.(x) .* y, dims = 1)
    y .- overlap .* x
end

"""
    random_pure(backend, T, d, batchsize)

Draw `batchsize` random pure states of dimension `d` on `backend`.

Returns a matrix of shape `(d, batchsize)` whose columns have unit norm.
"""
function random_pure(backend, ::Type{T}, d, batchsize) where {T}
    d = _validate_positive_int(:d, d)
    batchsize = _validate_positive_int(:batchsize, batchsize)
    _random_pure(backend, T, d, batchsize)
end

function _random_pure(::CPU, ::Type{T}, d, batchsize) where {T}
    ψd = randn(T, d, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims = 1)))
    ψd .*= norm_invs
    return ψd
end

"""
    random_overlap(backend, T, d, batchsize, q)

Generate pairs of states with controlled overlap parameter `q` on `backend`.

Returns `(xq, x)` where both have shape `(d, batchsize)`.
"""
function random_overlap(backend, ::Type{T}, d, batchsize, q::Real) where {T}
    d = _validate_positive_int(:d, d)
    batchsize = _validate_positive_int(:batchsize, batchsize)
    q = _validate_q(q)

    x = random_pure(backend, T, d, batchsize)
    y = random_pure(backend, T, d, batchsize)
    xp = gram_schmidt_step(backend, x, y)
    norm_xp = sqrt.(sum(real.(xp .* conj.(xp)), dims = 1))
    xp ./= norm_xp
    return sqrt(1 - q^2) * xp + q * x, x
end

"""
    random_unitary(backend, T, d, batchsize)

Draw `batchsize` random unitary matrices of size `d x d` on `backend`.

Returns an array of shape `(d, d, batchsize)`.
"""
function random_unitary(backend, ::Type{T}, d, batchsize) where {T}
    d = _validate_positive_int(:d, d)
    batchsize = _validate_positive_int(:batchsize, batchsize)
    _random_unitary(backend, T, d, batchsize)
end

@kernel inbounds = true function _reconstruct_q_from_compact_qr_kernel!(Q, A, tau_mat, phases, d, batchsize)
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

            diag_val = A[k, k, batch_idx]
            abs2_diag = abs2(diag_val)
            phases[k, batch_idx] = abs2_diag == 0 ? one(eltype(Q)) : diag_val / sqrt(abs2_diag)
        end
    end
end

@kernel function _apply_row_phases_kernel!(Q, phases, d, batchsize)
    batch_idx = @index(Global, Linear)

    if batch_idx <= batchsize
        @inbounds for row in 1:d
            @inbounds for col in 1:d
                Q[row, col, batch_idx] *= phases[row, batch_idx]
            end
        end
    end
end

function _reconstruct_q_from_compact_qr!(
    backend,
    Q,
    A,
    tau_mat,
    phases,
    d::Integer,
    batchsize::Integer;
    apply_phases::Bool = true,
)
    _reconstruct_q_from_compact_qr_kernel!(backend)(
        Q,
        A,
        tau_mat,
        phases,
        d,
        batchsize,
        ndrange = batchsize,
    )
    KernelAbstractions.synchronize(backend)

    if apply_phases
        _apply_row_phases_kernel!(backend)(Q, phases, d, batchsize, ndrange = batchsize)
        KernelAbstractions.synchronize(backend)
    end

    return Q
end

function _random_unitary(::CPU, ::Type{T}, d, batchsize) where {T}
    backend = CPU()
    Z = randn(T, d, d, batchsize)

    # geqrf! stores compact Householder data in-place in Z and returns τ.
    tau_mat = zeros(T, d, batchsize)
    for i in 1:batchsize
        tau_vec = zeros(T, d)
        LAPACK.geqrf!(view(Z, :, :, i), tau_vec)
        tau_mat[:, i] .= tau_vec
    end

    Q = zeros(T, d, d, batchsize)
    phases = zeros(T, d, batchsize)
    _reconstruct_q_from_compact_qr!(backend, Q, Z, tau_mat, phases, d, batchsize)
    return Q
end

"""
    random_max_ent(backend, T, d, batchsize)

Generate `batchsize` maximally entangled bipartite states of local dimension `d`
using random unitaries on `backend`.
"""
function random_max_ent(backend, ::Type{T}, d, batchsize) where {T}
    d = _validate_positive_int(:d, d)
    batchsize = _validate_positive_int(:batchsize, batchsize)

    Q = random_unitary(backend, T, d, batchsize)
    ψ = reduce(hcat, vec.(eachslice(Q; dims = 3))) ./ T(d)
    return ψ
end
