using ProgressMeter
using KernelAbstractions

function _validate_square_matrix(A::AbstractMatrix, name::AbstractString = "A")
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("`$name` must be square, got size $(size(A))"))
    return A
end

@kernel function batched_kron_kernel(z, @Const(x), @Const(y))
    i = @index(Global, Linear)
    if i <= size(z, 2)
        d1 = size(x, 1)
        d2 = size(y, 1)
        for k in 1:d1, l in 1:d2
            z[(k - 1) * d2 + l, i] = x[k, i] * y[l, i]
        end
    end
end

function batched_kron!(z, x, y)
    backend = KernelAbstractions.get_backend(z)
    kernel = batched_kron_kernel(backend)
    kernel(z, x, y, ndrange = size(z, 2))
end

"""
    shadow(backend, A, samples, sampling_f; batchsize=1_000_000)

Estimate the numerical shadow of matrix `A` on `backend`.

`sampling_f` is called as `sampling_f(backend, d, n)` and must return `n`
state vectors of dimension `d`.
"""
function shadow(backend, A::AbstractMatrix, samples, sampling_f, batchsize = 1_000_000)
    _validate_square_matrix(A)
    samples = _validate_positive_int(:samples, samples)
    batchsize = _validate_positive_int(:batchsize, batchsize)

    d = size(A, 1)
    Ad = move_to_backend(backend, A)
    dims_edges = get_bin_edges(A, 1000)
    x_edges = move_to_backend(backend, collect(dims_edges[1]))
    y_edges = move_to_backend(backend, collect(dims_edges[2]))
    
    shadow_hist = Hist2D(x_edges, y_edges)
    
    ranges = 1:batchsize:samples
    @showprogress for start_idx in ranges
        current_samples = min(batchsize, samples - start_idx + 1)
        # sampling_f must invoke generator on backend
        ψd = sampling_f(backend, d, current_samples)
        z = Ad * ψd
        conj!(ψd)
        points = vec(sum(z .* ψd, dims=1))
        shadow_hist += histogram(points, x_edges, y_edges)
    end
    return shadow_hist
end

"""
    product_qshadow(backend, T, A, samples, q; batchsize=1_000_000)

Estimate the product-state q-shadow for matrix `A` on `backend`.
"""
function product_qshadow(backend, ::Type{T}, A::AbstractMatrix, samples, q::Real, batchsize = 1_000_000) where {T}
    _validate_square_matrix(A)
    samples = _validate_positive_int(:samples, samples)
    batchsize = _validate_positive_int(:batchsize, batchsize)
    q = _validate_q(q)

    d = isqrt(size(A, 1))
    d * d == size(A, 1) ||
        throw(ArgumentError("`size(A, 1)` must be a perfect square, got $(size(A, 1))"))

    Ad = move_to_backend(backend, A)
    dims_edges = get_bin_edges(A, 1000, q)
    x_edges = move_to_backend(backend, collect(dims_edges[1]))
    y_edges = move_to_backend(backend, collect(dims_edges[2]))
    shadow_hist = Hist2D(x_edges, y_edges)
    
    ranges = 1:batchsize:samples
    @showprogress for start_idx in ranges
        current_samples = min(batchsize, samples - start_idx + 1)
        xq, x = random_overlap(backend, T, d, current_samples, sqrt(q))
        yq, y = random_overlap(backend, T, d, current_samples, sqrt(q))
        
        zq = KernelAbstractions.zeros(backend, T, d*d, current_samples)
        z = KernelAbstractions.zeros(backend, T, d*d, current_samples)
        
        batched_kron!(zq, xq, yq)
        batched_kron!(z, x, y)
        
        p = Ad * z
        conj!(zq)
        points = vec(sum(p .* zq, dims=1))
        shadow_hist += histogram(points, x_edges, y_edges)
    end
    return shadow_hist
end

"""
    qshadow(backend, T, A, samples, q; batchsize=1_000_000)

Estimate the q-shadow of matrix `A` on `backend`.
"""
function qshadow(backend, ::Type{T}, A::AbstractMatrix, samples, q::Real, batchsize = 1_000_000) where {T}
    _validate_square_matrix(A)
    samples = _validate_positive_int(:samples, samples)
    batchsize = _validate_positive_int(:batchsize, batchsize)
    q = _validate_q(q)

    d = size(A, 1)
    Ad = move_to_backend(backend, A)
    dims_edges = get_bin_edges(A, 1000, q)
    x_edges = move_to_backend(backend, collect(dims_edges[1]))
    y_edges = move_to_backend(backend, collect(dims_edges[2]))
    shadow_hist = Hist2D(x_edges, y_edges)
    
    ranges = 1:batchsize:samples
    @showprogress for start_idx in ranges
        current_samples = min(batchsize, samples - start_idx + 1)
        zq, z = random_overlap(backend, T, d, current_samples, q)
        p = Ad * z
        conj!(zq)
        points = vec(sum(p .* zq, dims=1))
        shadow_hist += histogram(points, x_edges, y_edges)
    end
    return shadow_hist
end
