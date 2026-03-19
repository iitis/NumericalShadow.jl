using KernelAbstractions
using LinearAlgebra
using Random
using Test

function _shadow_test_backends()
    backends = Any[CPU()]
    if HAS_CUDA
        push!(backends, CUDABackend())
    end
    return backends
end

@testset verbose = true "Shadow Workflow" begin
    Random.seed!(1234)

    A = [1.0 0.0; 0.0 -1.0]
    samples = 100
    sampler(b, d, n) = random_pure(b, ComplexF64, d, n)

    for backend in _shadow_test_backends()
        backend_name = backend isa CPU ? "CPU" : "CUDA"
        @testset "Backend: $backend_name" begin
            h = NumericalShadow.shadow(backend, A, samples, sampler)
            @test h isa NumericalShadow.Hist2D
            @test sum(collect(h.hist)) == samples

            h_q = NumericalShadow.qshadow(backend, ComplexF64, A, samples, 0.5)
            @test sum(collect(h_q.hist)) == samples

            A4 = Matrix(I, 4, 4)
            h_pq = NumericalShadow.product_qshadow(backend, ComplexF64, A4, samples, 0.5)
            @test sum(collect(h_pq.hist)) == samples
        end
    end
end
