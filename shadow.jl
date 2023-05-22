using Plots
using LinearAlgebra
using MatrixEnsembles, QuantumInformation
using BenchmarkTools
using Base.Threads
using Flux, MLUtils
using ProgressMeter
using CUDA

function random_pure(d, batchsize)
    ψd = CUDA.randn(ComplexF32, d, 1, batchsize)
    norm_invs = 1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims=1))
    ψd = ψd ⊠ norm_invs;
    return ψd
end

function shadow_GPU(A::Matrix, samples::Int, batchsize::Int=1_000_000)
    num_batches = div(samples, batchsize)
    d = size(A, 1)
    Ad = cu(unsqueeze(A, 3))
    @showprogress for i=1:num_batches
        ψd = random_pure(d, batchsize)
        batched_adjoint(ψd) ⊠ Ad ⊠ ψd
    end
end

function shadow_batched(A::Matrix, samples::Int, batchsize::Int=1_000_000)
    num_batches = div(samples, batchsize)
    d = size(A, 1)
    Ad = unsqueeze(A, 3)
    dist = HaarKet(d)
    @showprogress for i=1:num_batches
        ψd = batch((unsqueeze(rand(dist), 2) for _ = 1:batchsize))
        batched_adjoint(ψd) ⊠ Ad ⊠ ψd
    end
end

function plot_shadow(s::Vector{ComplexF64}, A, fname)
    λ = eigvals(A)
    θ = minimum(diff(abs.(angle.(λ)))) / 2
    d = size(A, 1)
    angles = [exp(1im * θ * i) for i=1:2d] ./ 4
    p = histogram2d(real.(s), imag.(s), nbins=1000, aspect_ratio=:equal, legend=false)
    plot!(p, aspect_ratio=:equal)
    plot!(p, framestyle=:box)
    plot!(p, size=(400, 400))
    plot!(p, real(λ), imag(λ), seriestype=:scatter, color=:red, markersize=4)
    plot!(p, real(angles), imag(angles), seriestype=:scatter, color=:black, markersize=4)
    savefig(p, fname)
    return p
end

function plot_shadow(s::Vector{Float64}, A, fname)
    λ = eigvals(A)
    histogram(s, nbins=100, legend=false)
    plot!(xlims=(-1, 1), ylims=(0, 1), aspect_ratio=:equal)
    plot!(xaxis=false, yaxis=false)
    plot!(framestyle=:box)
    plot!(size=(400, 400))
    savefig(fname)
end

function max_entangled_shadow(A::Matrix, samples::Int)
    # Generate a random pure state
    d = isqrt(size(A, 1))
    ret = zeros(ComplexF64, samples)
    # Generate a random unitary matrix
    dist = CUE(d)
    Threads.@threads for i=1:samples
        U = rand(dist)
        ψ = vec(U)
        ret[i] = ψ' * A * ψ
    end
    return ret
end

function shadow(A::Matrix, samples::Int)
    d = size(A, 1)
    dist = HaarKet(d)
    ret = zeros(ComplexF64, samples)
    Threads.@threads for i=1:samples
        ψ = rand(dist)
        ret[i] = ψ' * A * ψ
    end
    return ret
end

function shadow2(A::Matrix, samples::Int, batchsize::Int=1_000_000)
    d = size(A, 1)
    dist = HaarKet(d)
    ret = zeros(ComplexF64, samples)
    num_batches = div(samples, batchsize)
    d = size(A, 1)
    Threads.@threads for i=1:num_batches
        for i=1:samples
            ψ = rand(dist)
            #ret[i] =
             ψ' * A * ψ
        end
    end
    return ret
end

function max_entangled_shadow(A::Hermitian, samples::Int)
    ret = zeros(samples)
    d = isqrt(size(A, 1))
    dist = CUE(d)
    Threads.@threads for i=1:samples
        U = rand(dist)
        ψ = vec(U)
        ret[i] = ψ' * A * ψ
    end
    return ret
end

function compare_shadows(samples, n)

    A = diagm(0=>[exp(2 * pi * 1im * i / n ) for i=1:n])
    @time ret = max_entangled_shadow(A, samples)
    p1 = plot_shadow(ret, A, "ent_shadow_complex.png")

    @time ret = shadow(A, samples)
    p2 = plot_shadow(ret, A, "shadow_complex.png")

    l = @layout [a b]
    plot(p1, p2, layout=l)
    savefig("complex.png")
end
# BLAS.set_num_threads(1)
n = 9
samples = 100_000_000
A = diagm(0=>[exp(2 * pi * 1im * i / n ) for i=1:n])
compare_shadows(samples, n)