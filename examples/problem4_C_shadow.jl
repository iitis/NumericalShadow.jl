# Shadow.pdf mapping:
#   Problem 4 (C-numerical range shadow)
#
# We sample Haar-random unitary U and collect points:
#   tr(C * U * A * U')
# corresponding to W_C(A) = {tr(C U A U*) : U*U = I}.

using NumericalShadow
using KernelAbstractions
using LinearAlgebra
using CUDA
using ProgressMeter
using Random

function choose_backend()
    use_cuda = lowercase(get(ENV, "NUMERICALSHADOW_USE_CUDA", "true")) in ("1", "true", "yes")
    if use_cuda && CUDA.functional()
        println("Using CUDA backend.")
        return CUDABackend()
    end
    println("Using CPU backend.")
    return CPU()
end

function edges_from_points(points::AbstractVector{<:Complex}; nbins::Int = 1000, pad_frac::Real = 0.05)
    nbins > 0 || throw(ArgumentError("`nbins` must be positive, got $nbins"))
    xs = real.(points)
    ys = imag.(points)
    min_x, max_x = extrema(xs)
    min_y, max_y = extrema(ys)

    dx = max(max_x - min_x, 1e-8)
    dy = max(max_y - min_y, 1e-8)

    min_x -= pad_frac * dx
    max_x += pad_frac * dx
    min_y -= pad_frac * dy
    max_y += pad_frac * dy

    x_edges = collect(range(min_x, max_x; length = nbins + 1))
    y_edges = collect(range(min_y, max_y; length = nbins + 1))
    return x_edges, y_edges
end

function sample_C_shadow_points(
    backend,
    A::AbstractMatrix{<:Complex},
    C::AbstractMatrix{<:Complex},
    samples::Int,
    batchsize::Int,
    ::Type{T} = ComplexF64,
) where {T}
    size(A, 1) == size(A, 2) || throw(ArgumentError("`A` must be square"))
    size(C, 1) == size(C, 2) || throw(ArgumentError("`C` must be square"))
    size(A, 1) == size(C, 1) || throw(ArgumentError("`size(A)` and `size(C)` must match"))

    n = size(A, 1)
    points = Vector{ComplexF64}(undef, samples)
    idx = 1

    @showprogress "Sampling Problem 4 points" for start_idx in 1:batchsize:samples
        current_batch = min(batchsize, samples - start_idx + 1)
        U_batch = NumericalShadow.random_unitary(backend, T, n, current_batch)
        U_cpu = Array(U_batch)

        for i in 1:current_batch
            Ui = view(U_cpu, :, :, i)
            points[idx] = tr(C * Ui * A * adjoint(Ui))
            idx += 1
        end
    end

    return points
end

backend = choose_backend()
Random.seed!(42)

samples = parse(Int, get(ENV, "NUMERICALSHADOW_SAMPLES", "50000000"))
batchsize = parse(Int, get(ENV, "NUMERICALSHADOW_BATCHSIZE", "500000"))
nbins = parse(Int, get(ENV, "NUMERICALSHADOW_NBINS", "1000"))

n = 6
A = randn(ComplexF64, n, n)
C = Matrix(Diagonal(ComplexF64[2.0, 1.0, 0.0, -0.5, -1.0, -1.5]))
C[1, 2] = 0.6 + 0.2im
C[3, 5] = -0.4im

points = sample_C_shadow_points(backend, A, C, samples, batchsize, ComplexF64)
x_edges, y_edges = edges_from_points(points; nbins = nbins)
h = NumericalShadow.histogram(points, x_edges, y_edges)
h.nr = NumericalShadow.numerical_range(A)
h.evs = eigvals(A)

star_center = tr(C) * tr(A) / n
println("Theoretical star center (Problem 4): ", star_center)

mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(@__DIR__, "results", "problem4_C_shadow_n=$(n).h5")
NumericalShadow.save(h, outpath)
println("Saved: $outpath")
