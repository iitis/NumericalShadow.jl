using NumericalShadow
using Test
using Aqua
using KernelAbstractions
using LinearAlgebra
using StatsBase

# Try to load CUDA for testing
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset verbose = true "NumericalShadow.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(NumericalShadow; ambiguities=false)
    end

    backends = Any[CPU()]
    if HAS_CUDA
        push!(backends, CUDABackend())
    end

    for backend in backends
        backend_name = backend isa CPU ? "CPU" : "CUDA"
        @testset verbose = true "Backend: $backend_name" begin
            
            @testset "Random Helpers" begin
                d = 4
                batch = 10
                T = ComplexF64
                
                # random_pure
                ψ = random_pure(backend, T, d, batch)
                @test size(ψ) == (d, batch)
                # Use collect to ensure we can check on CPU
                norms = sum(abs2.(collect(ψ)), dims=1)
                @test all(isapprox.(norms, 1.0, atol=1e-5))
                
                # random_unitary
                U = random_unitary(backend, T, d, batch)
                @test size(U) == (d, d, batch)
                U_cpu = collect(U)
                for i in 1:batch
                    u = U_cpu[:, :, i]
                    @test isapprox(u' * u, I, atol=1e-5)
                end
                
                # random_overlap
                q = 0.5
                x, x_orig = random_overlap(backend, T, d, batch, q)
                @test size(x) == (d, batch)
            end
            
            @testset "Shadow Workflow" begin
                A = [1.0 0.0; 0.0 -1.0]
                samples = 100
                sampler(b, d, n) = random_pure(b, ComplexF64, d, n)
                
                h = NumericalShadow.shadow(backend, A, samples, sampler)
                @test h isa NumericalShadow.Hist2D
                @test sum(collect(h.hist)) == samples
                
                # qshadow
                h_q = NumericalShadow.qshadow(backend, ComplexF64, A, samples, 0.5)
                @test sum(collect(h_q.hist)) == samples
                
                A4 = Matrix(I, 4, 4)
                h_pq = NumericalShadow.product_qshadow(backend, ComplexF64, A4, samples, 0.5)
                @test sum(collect(h_pq.hist)) == samples
            end
            @testset "Eigenvalue Distribution" begin
                # CUE test inspired by user request
                d_val = 100
                batch_val = 100
                
                # Generate a batch of random unitaries
                # We move to CPU for eigenvalue calc as generic eigvals might not be available/fast on GPU
                U_batch = random_unitary(backend, ComplexF64, d_val, batch_val)
                U_cpu = collect(U_batch)
                
                # Collect all eigenphases
                phases = Vector{Float64}()
                for i in 1:batch_val
                    vals = eigvals(U_cpu[:, :, i])
                    append!(phases, angle.(vals))
                end
                
                # Check distribution against uniform 1/2π
                # Using 20 bins from -π to π
                h = fit(Histogram, phases, -π:0.1π:π, closed=:left)
                h_norm = normalize(h, mode=:pdf)
                
                # Check if all bins are approximately 1/2π
                # 1/2π ≈ 0.159
                @test all(isapprox.(h_norm.weights, 1/(2π), atol=0.05))
            end
        end
    end
end
