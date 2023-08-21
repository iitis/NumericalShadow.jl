using ProgressMeter

function LinearAlgebra.kron!(z::CuMatrix, x::CuMatrix, y::CuMatrix)
    function kernel(z, x, y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i <= size(z, 2)
            d1 = size(x, 1)
            d2 = size(y, 1)
            for k=1:d1, l=1:d2
                z[(k-1)*d2+l, i] = x[k, i] * y[l, i]
            end
        end
        return
    end
    @assert size(z, 2) == size(x, 2) == size(y, 2)
    @assert size(z, 1) == size(x, 1) * size(y, 1)
    threads = 512
    blocks = cld(size(x, 2), threads)
    @cuda threads=threads blocks=blocks kernel(z, x, y)
end

function shadow_GPU(A::Matrix, samples::Int, sampling_f, batchsize::Int = 1_000_000)
    num_batches = div(samples, batchsize)
    d = size(A, 1)
    Ad = cu(A)
    x_edges, y_edges = cu.(collect.(get_bin_edges(A, 1000)))
    shadow = Hist2D(x_edges, y_edges)
    @showprogress for i = 1:num_batches
        ψd = sampling_f(d, num_batches == 1 ? samples : batchsize)
        z = Ad * ψd
        conj!(ψd)
        points = vec(sum(z .* ψd, dims=1))
        shadow += histogram(real(points), imag(points), x_edges, y_edges)
    end
    return shadow
end

function product_qshadow_GPU(::Type{T}, A::Matrix, samples::Int, q::Real, batchsize::Int = 1_000_000) where {T}
    num_batches = div(samples, batchsize)
    d = isqrt(size(A, 1))
    Ad = cu(A)
    x_edges, y_edges = cu.(collect.(get_bin_edges(A, 1000, q)))
    shadow = Hist2D(x_edges, y_edges)
    @showprogress for i = 1:num_batches
        xq, x = random_overlap(T, d, num_batches == 1 ? samples : batchsize, sqrt(q))
        yq, y = random_overlap(T, d, num_batches == 1 ? samples : batchsize, sqrt(q))
        zq = CUDA.zeros(T, d*d, batchsize)
        z = CUDA.zeros(T, d*d, batchsize)
        kron!(zq, xq, yq)
        kron!(z, x, y)
        p = Ad * z
        conj!(zq)
        points = vec(sum(p .* zq, dims=1))
        shadow += histogram(real(points), imag(points), x_edges, y_edges)
    end
    return shadow
end

function qshadow_GPU(::Type{T}, A::Matrix, samples::Int, q::Real, batchsize::Int = 1_000_000) where {T}
    num_batches = div(samples, batchsize)
    d = size(A, 1)
    Ad = cu(A)
    x_edges, y_edges = cu.(collect.(get_bin_edges(A, 1000, q)))
    shadow = Hist2D(x_edges, y_edges)
    @showprogress for i = 1:num_batches
        zq, z = random_overlap(T, d, num_batches == 1 ? samples : batchsize, q)
        p = Ad * z
        conj!(zq)
        points = vec(sum(p .* zq, dims=1))
        shadow += histogram(real(points), imag(points), x_edges, y_edges)
    end
    return shadow
end

# function max_entangled_shadow(A::Matrix, samples::Int)
#     # Generate a random pure state
#     d = isqrt(size(A, 1))
#     ret = zeros(ComplexF64, samples)
#     # Generate a random unitary matrix
#     dist = CUE(d)
#     Threads.@threads for i=1:samples
#         g = randn(ComplexF64, d, d)
#         q, r = LinearAlgebra.qr(g)
#         ψ = vec(collect(q))
#         # U = rand(dist)
#         # ψ = vec(U)
#         ret[i] = ψ' * A * ψ
#     end
#     return ret
# end

# function f(dist)
#     ψ = rand(dist)
#     return ψ' * A * ψ
# end

# function shadow(A::Matrix, samples::Int)
#     d = size(A, 1)
#     dist = HaarKet(d)
#     ret = zeros(ComplexF64, samples)

#     Threads.@threads for i=1:samples
#         ret[i] = f(dist)
#     end
#     return ret
# end

# function shadow2(A::Matrix, samples::Int, batchsize::Int=1_000_000)
#     d = size(A, 1)
#     dist = HaarKet(d)
#     ret = zeros(ComplexF64, samples)
#     num_batches = div(samples, batchsize)
#     d = size(A, 1)
#     Threads.@threads for i=1:num_batches
#         for i=1:samples
#             ψ = rand(dist)
#             ret[i] = ψ' * A * ψ
#         end
#     end
#     return ret
# end
