using KernelAbstractions
using LinearAlgebra
using Random
using Test

function _available_backends()
    backends = Any[CPU()]
    if HAS_CUDA
        push!(backends, CUDABackend())
    end
    return backends
end

@testset verbose = true "Random Helpers" begin
    Random.seed!(1234)

    d = 4
    batch = 10
    T = ComplexF64

    for backend in _available_backends()
        backend_name = backend isa CPU ? "CPU" : "CUDA"
        @testset "Backend: $backend_name" begin
            ψ = random_pure(backend, T, d, batch)
            @test size(ψ) == (d, batch)
            norms = sum(abs2.(collect(ψ)), dims = 1)
            @test all(isapprox.(norms, 1.0, atol = 1e-5))

            U = random_unitary(backend, T, d, batch)
            @test size(U) == (d, d, batch)
            U_cpu = collect(U)
            for i in 1:batch
                u = U_cpu[:, :, i]
                @test isapprox(u' * u, I, atol = 1e-5)
            end

            q = 0.5
            x, _ = random_overlap(backend, T, d, batch, q)
            @test size(x) == (d, batch)
        end
    end
end
