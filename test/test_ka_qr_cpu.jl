using KernelAbstractions
using LinearAlgebra
using NumericalShadow
using Random
using Test

const HAS_CUDA_KA_QR_TEST = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

const RUN_SLOW_TESTS = lowercase(get(ENV, "NUMERICALSHADOW_RUN_SLOW_TESTS", "false")) in
                       ("1", "true", "yes")

function native_random_unitary(::Type{T}, d::Int, batchsize::Int) where {T}
    Q = zeros(T, d, d, batchsize)
    for i in 1:batchsize
        g = randn(T, d, d)
        q_val, r_val = qr(g)
        d_diag = sign.(diag(r_val))
        d_diag[d_diag .== 0] .= 1
        Q[:, :, i] .= Matrix(q_val) * Diagonal(d_diag)
    end
    return Q
end

function ka_reconstruct_from_geqrf(g::AbstractMatrix{T}; apply_phases::Bool) where {T}
    d = size(g, 1)
    Z = copy(g)
    tau_vec = zeros(T, d)
    LAPACK.geqrf!(Z, tau_vec)

    Q = zeros(T, d, d, 1)
    tau_mat = reshape(tau_vec, d, 1)
    phases = zeros(T, d, 1)
    Z3d = reshape(Z, d, d, 1)

    NumericalShadow._reconstruct_q_from_compact_qr!(
        CPU(),
        Q,
        Z3d,
        tau_mat,
        phases,
        d,
        1;
        apply_phases = apply_phases,
    )
    return Q[:, :, 1], phases[:, 1]
end

function collect_eigenphases(Q_batch)
    d1, d2, batchsize = size(Q_batch)
    @assert d1 == d2
    phases = Vector{Float64}(undef, d1 * batchsize)
    for i in 1:batchsize
        sample_range = ((i - 1) * d1 + 1):(i * d1)
        phases[sample_range] .= angle.(eigvals(Q_batch[:, :, i]))
    end
    return phases
end

function phase_density(phases::AbstractVector{<:Real})
    bin_edges = collect(-π:0.1π:π)
    n_bins = length(bin_edges) - 1
    bin_width = bin_edges[2] - bin_edges[1]
    counts = zeros(Int, n_bins)

    for θ in phases
        idx = searchsortedlast(bin_edges, θ)
        if idx == length(bin_edges)
            # Keep θ == π in the last bin.
            idx -= 1
        end
        if 1 <= idx <= n_bins
            counts[idx] += 1
        end
    end

    return counts ./ (length(phases) * bin_width)
end

@testset "KA QR vs Native QR on CPU" begin
    Random.seed!(1234)

    T = ComplexF64
    d = 6
    batchsize = 100

    @testset "KA produces unitary matrices" begin
        Q_ka = random_unitary(CPU(), T, d, batchsize)
        for i in 1:batchsize
            Qi = Q_ka[:, :, i]
            @test Qi' * Qi ≈ I atol = 1e-10
            @test Qi * Qi' ≈ I atol = 1e-10
            @test abs(det(Qi)) ≈ 1.0 atol = 1e-10
        end
    end

    @testset "Native produces unitary matrices" begin
        Q_nat = native_random_unitary(T, d, batchsize)
        for i in 1:batchsize
            Qi = Q_nat[:, :, i]
            @test Qi' * Qi ≈ I atol = 1e-10
            @test Qi * Qi' ≈ I atol = 1e-10
            @test abs(det(Qi)) ≈ 1.0 atol = 1e-10
        end
    end

    @testset "Householder reconstruction matches native Q (before phases)" begin
        # The raw Q from Householder reconstruction should match Julia's qr() Q factor.
        for _ in 1:20
            g = randn(T, d, d)
            q_val, _ = qr(g)
            Q_native = Matrix(q_val)

            Q_ka_raw, _ = ka_reconstruct_from_geqrf(g; apply_phases = false)
            @test Q_ka_raw ≈ Q_native atol = 1e-10
        end
    end

    @testset "Phase conventions: KA uses row scaling, native uses column scaling" begin
        for _ in 1:10
            g = randn(T, d, d)

            q_val, r_val = qr(g)
            d_diag = sign.(diag(r_val))
            d_diag[d_diag .== 0] .= 1
            Q_native_phased = Matrix(q_val) * Diagonal(d_diag)

            Q_ka, phase_vec = ka_reconstruct_from_geqrf(g; apply_phases = true)

            # Q_ka = Diagonal(phase) * Q_native_raw
            # Q_native_phased = Q_native_raw * Diagonal(phase)
            # so Q_ka = Diagonal(phase) * Q_native_phased * Diagonal(conj.(phase))
            expected = Diagonal(phase_vec) * Q_native_phased * Diagonal(conj.(phase_vec))
            @test Q_ka ≈ expected atol = 1e-10
            @test Q_ka' * Q_ka ≈ I atol = 1e-10
            @test Q_native_phased' * Q_native_phased ≈ I atol = 1e-10
        end
    end

    @testset "Singular values are all 1" begin
        Q_ka = random_unitary(CPU(), T, d, batchsize)
        for i in 1:min(20, batchsize)
            sv = svdvals(Q_ka[:, :, i])
            @test all(isapprox.(sv, 1.0, atol = 1e-10))
        end
    end

    @testset "Eigenvalues on unit circle" begin
        Q_ka = random_unitary(CPU(), T, d, batchsize)
        for i in 1:min(20, batchsize)
            evs = eigvals(Q_ka[:, :, i])
            @test all(isapprox.(abs.(evs), 1.0, atol = 1e-10))
        end
    end

    @testset "Eigenvalue phase distribution is approximately uniform" begin
        if !RUN_SLOW_TESTS
            @test_skip "Set NUMERICALSHADOW_RUN_SLOW_TESTS=true to enable statistical phase tests."
        else
            # CUE-like marginal check: eigenphases should be approximately uniform on [-π, π).
            dim = 100
            steps = 100
            expected_density = 1 / (2π)

            for Q_batch in (
                random_unitary(CPU(), T, dim, steps),
                native_random_unitary(T, dim, steps),
            )
                phases = collect_eigenphases(Q_batch)
                empirical_density = phase_density(phases)
                @test all(isapprox.(empirical_density, expected_density, atol = 0.03))
            end
        end
    end

    @testset "Different matrix sizes" begin
        for dim in [2, 3, 5, 8, 12]
            Q_ka = random_unitary(CPU(), T, dim, 10)
            for i in 1:10
                Qi = Q_ka[:, :, i]
                @test Qi' * Qi ≈ I atol = 1e-10
            end
        end
    end

    @testset "ComplexF32 support" begin
        Q_ka = random_unitary(CPU(), ComplexF32, d, 20)
        for i in 1:20
            Qi = Q_ka[:, :, i]
            @test Qi' * Qi ≈ I atol = 1e-4
            @test abs(det(Qi)) ≈ 1.0 atol = 1e-4
        end
    end
end

@testset "KA QR CUDA path (if available)" begin
    if !HAS_CUDA_KA_QR_TEST
        @test_skip "CUDA not functional in this environment"
    else
        Random.seed!(1234)

        backend = CUDABackend()
        T = ComplexF64
        d = 6
        batchsize = 40

        Q_gpu = random_unitary(backend, T, d, batchsize)
        Q = Array(Q_gpu)

        @test size(Q) == (d, d, batchsize)
        for i in 1:batchsize
            Qi = Q[:, :, i]
            @test Qi' * Qi ≈ I atol = 1e-8
            @test Qi * Qi' ≈ I atol = 1e-8
            @test abs(det(Qi)) ≈ 1.0 atol = 1e-8
        end

        @testset "Eigenvalue phase distribution is approximately uniform" begin
            if !RUN_SLOW_TESTS
                @test_skip "Set NUMERICALSHADOW_RUN_SLOW_TESTS=true to enable statistical phase tests."
            else
                dim = 64
                steps = 40
                expected_density = 1 / (2π)

                Q_batch = Array(random_unitary(backend, T, dim, steps))
                phases = collect_eigenphases(Q_batch)
                empirical_density = phase_density(phases)
                @test all(isapprox.(empirical_density, expected_density, atol = 0.06))
            end
        end
    end
end
