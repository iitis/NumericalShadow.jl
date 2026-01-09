export shadow_GPU
using ProgressMeter
using KernelAbstractions

@kernel function batched_kron_kernel(z, @Const(x), @Const(y))
    i = @index(Global, Linear)
    if i <= size(z, 2)
        d1 = size(x, 1)
        d2 = size(y, 1)
        for k=1:d1, l=1:d2
            z[(k-1)*d2+l, i] = x[k, i] * y[l, i]
        end
    end
end

function batched_kron!(z, x, y)
    backend = KernelAbstractions.get_backend(z)
    kernel = batched_kron_kernel(backend)
    kernel(z, x, y, ndrange=size(z, 2))
end

function shadow(backend, A::Matrix, samples::Int, sampling_f, batchsize::Int = 1_000_000)
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

function product_qshadow(backend, ::Type{T}, A::Matrix, samples::Int, q::Real, batchsize::Int = 1_000_000) where {T}
    d = isqrt(size(A, 1))
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

function qshadow(backend, ::Type{T}, A::Matrix, samples::Int, q::Real, batchsize::Int = 1_000_000) where {T}
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
