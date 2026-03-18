using KernelAbstractions
using LinearAlgebra
using Test

# Copy the kernels from the CUDA extension, but run on CPU
@kernel inbounds = true function reconstruct_Q_unitary!(Q, A, tau_mat, phases, d, batchsize)
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

            phases[k, batch_idx] = A[k, k, batch_idx] / sqrt(abs2(A[k, k, batch_idx]))
        end
    end
end

@kernel function apply_phases!(Q, phases, d, batchsize)
    batch_idx = @index(Global, Linear)

    if batch_idx <= batchsize
        @inbounds for row in 1:d
            @inbounds for col in 1:d
                Q[row, col, batch_idx] *= phases[row, batch_idx]
            end
        end
    end
end

"""
    ka_random_unitary(T, d, batchsize)

Generate random unitary matrices using LAPACK geqrf! + KernelAbstractions kernels on CPU.
This mirrors the CUDA extension path but runs entirely on CPU.
"""
function ka_random_unitary(::Type{T}, d::Int, batchsize::Int) where {T}
    backend = CPU()

    Z = randn(T, d, d, batchsize)

    # Use LAPACK geqrf! on each slice to get compact Householder form + tau
    tau_mat = zeros(T, d, batchsize)
    for i in 1:batchsize
        slice = view(Z, :, :, i)
        tau_vec = zeros(T, d)
        LAPACK.geqrf!(slice, tau_vec)
        tau_mat[:, i] .= tau_vec
    end

    Q = zeros(T, d, d, batchsize)
    phases = zeros(T, d, batchsize)

    kernel = reconstruct_Q_unitary!(backend)
    kernel(Q, Z, tau_mat, phases, d, batchsize, ndrange=batchsize)
    KernelAbstractions.synchronize(backend)

    apply_phases!(backend)(Q, phases, d, batchsize, ndrange=batchsize)
    KernelAbstractions.synchronize(backend)

    return Q
end

"""
    native_random_unitary(T, d, batchsize)

Generate random unitary matrices using Julia's native qr() with phase correction.
This is the existing CPU path from NumericalShadow.jl.
"""
function native_random_unitary(::Type{T}, d::Int, batchsize::Int) where {T}
    Q = zeros(T, d, d, batchsize)
    for i in 1:batchsize
        g = randn(T, d, d)
        q_val, r_val = qr(g)
        d_diag = sign.(diag(r_val))
        d_diag[d_diag.==0] .= 1
        d_mat = Diagonal(d_diag)
        Q[:, :, i] .= Matrix(q_val) * d_mat
    end
    return Q
end

@testset "KA QR vs Native QR on CPU" begin
    T = ComplexF64
    d = 6
    batchsize = 100

    @testset "KA produces unitary matrices" begin
        Q_ka = ka_random_unitary(T, d, batchsize)
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
        # The raw Q from Householder reconstruction should exactly match
        # Julia's qr() Q factor. The phase correction differs:
        #   Native: Q * Diagonal(sign(diag(R)))  (column scaling)
        #   KA:     Diagonal(sign(diag(R))) * Q  (row scaling)
        # Both produce valid Haar-random unitaries since left/right multiplication
        # by a diagonal unitary preserves the Haar measure.
        for _ in 1:20
            g = randn(T, d, d)

            # Native Q (no phase correction)
            q_val, _ = qr(g)
            Q_native = Matrix(q_val)

            # KA path: reconstruct Q from geqrf! output (no phase step)
            Z = copy(g)
            tau_vec = zeros(T, d)
            LAPACK.geqrf!(Z, tau_vec)

            Q_ka = zeros(T, d, d, 1)
            tau_mat = reshape(tau_vec, d, 1)
            phases = zeros(T, d, 1)
            Z3d = reshape(Z, d, d, 1)

            backend = CPU()
            reconstruct_Q_unitary!(backend)(Q_ka, Z3d, tau_mat, phases, d, 1, ndrange=1)
            KernelAbstractions.synchronize(backend)
            # Do NOT apply phases — compare raw Q

            @test Q_ka[:, :, 1] ≈ Q_native atol = 1e-10
        end
    end

    @testset "Phase conventions: KA uses row scaling, native uses column scaling" begin
        for _ in 1:10
            g = randn(T, d, d)

            # Native: Q * Diagonal(sign(diag(R)))
            q_val, r_val = qr(g)
            d_diag = sign.(diag(r_val))
            d_diag[d_diag.==0] .= 1
            Q_native_phased = Matrix(q_val) * Diagonal(d_diag)

            # KA: Diagonal(sign(diag(R))) * Q  (row scaling via apply_phases! kernel)
            Z = copy(g)
            tau_vec = zeros(T, d)
            LAPACK.geqrf!(Z, tau_vec)

            Q_ka = zeros(T, d, d, 1)
            tau_mat = reshape(tau_vec, d, 1)
            phases = zeros(T, d, 1)
            Z3d = reshape(Z, d, d, 1)

            backend = CPU()
            reconstruct_Q_unitary!(backend)(Q_ka, Z3d, tau_mat, phases, d, 1, ndrange=1)
            KernelAbstractions.synchronize(backend)
            apply_phases!(backend)(Q_ka, phases, d, 1, ndrange=1)
            KernelAbstractions.synchronize(backend)

            # Verify: Q_ka = Diagonal(phases) * Q_native_raw
            # and Q_native_phased = Q_native_raw * Diagonal(phases)
            # so Q_ka = Diagonal(phases) * Q_native_phased * Diagonal(conj.(phases))
            phase_vec = phases[:, 1]
            expected = Diagonal(phase_vec) * Q_native_phased * Diagonal(conj.(phase_vec))
            @test Q_ka[:, :, 1] ≈ expected atol = 1e-10

            # Both are still unitary
            @test Q_ka[:, :, 1]' * Q_ka[:, :, 1] ≈ I atol = 1e-10
            @test Q_native_phased' * Q_native_phased ≈ I atol = 1e-10
        end
    end

    @testset "Singular values are all 1" begin
        Q_ka = ka_random_unitary(T, d, batchsize)
        for i in 1:min(20, batchsize)
            sv = svdvals(Q_ka[:, :, i])
            @test all(isapprox.(sv, 1.0, atol=1e-10))
        end
    end

    @testset "Eigenvalues on unit circle" begin
        Q_ka = ka_random_unitary(T, d, batchsize)
        for i in 1:min(20, batchsize)
            evs = eigvals(Q_ka[:, :, i])
            @test all(isapprox.(abs.(evs), 1.0, atol=1e-10))
        end
    end

    @testset "Different matrix sizes" begin
        for dim in [2, 3, 5, 8, 12]
            Q_ka = ka_random_unitary(T, dim, 10)
            for i in 1:10
                Qi = Q_ka[:, :, i]
                @test Qi' * Qi ≈ I atol = 1e-10
            end
        end
    end

    @testset "ComplexF32 support" begin
        Q_ka = ka_random_unitary(ComplexF32, d, 20)
        for i in 1:20
            Qi = Q_ka[:, :, i]
            @test Qi' * Qi ≈ I atol = 1e-4
            @test abs(det(Qi)) ≈ 1.0 atol = 1e-4
        end
    end
end
