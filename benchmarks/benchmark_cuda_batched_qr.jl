using BenchmarkTools
using CUDA
using KernelAbstractions
using LinearAlgebra
using NumericalShadow
using Printf

const D = 9
const BATCH_SIZE = 10_000
const T = ComplexF64

const BENCH_EVALS = 1
const BENCH_SAMPLES = 15
const BENCH_SECONDS = 15.0

struct CaseStats
    median_ms::Float64
    mean_ms::Float64
end

function ensure_cuda!()
    if !CUDA.functional()
        println("CUDA is not functional in this environment. Aborting benchmark.")
        exit(1)
    end
end

function phase_corrected_q_from_qr_slice(A::CuArray{T, 2}) where {T}
    F = qr(A)
    Q_part = CuArray(F.Q)
    R_part = CuArray(F.R)
    phase_vec = sign.(diag(R_part))
    phase_vec = ifelse.(phase_vec .== zero(T), one(T), phase_vec)
    return Q_part * Diagonal(phase_vec)
end

function batched_q_from_compact_qr(work::CuArray{T, 3}, backend::CUDABackend) where {T}
    d = size(work, 1)
    batch_size = size(work, 3)
    tau, _ = CUDA.CUBLAS.geqrf_batched!([view(work, :, :, i) for i in 1:batch_size])

    tau_mat = CUDA.zeros(T, d, batch_size)
    phases = CUDA.zeros(T, d, batch_size)
    Q = CUDA.zeros(T, d, d, batch_size)

    for i in 1:batch_size
        tau_mat[:, i] .= tau[i]
    end

    NumericalShadow._reconstruct_q_from_compact_qr!(backend, Q, work, tau_mat, phases, d, batch_size)
    return Q
end

function looped_q_from_qr(work::CuArray{T, 3}) where {T}
    d = size(work, 1)
    batch_size = size(work, 3)
    Q = CUDA.zeros(T, d, d, batch_size)
    for i in 1:batch_size
        Q[:, :, i] .= phase_corrected_q_from_qr_slice(view(work, :, :, i))
    end
    return Q
end

function batched_end_to_end(backend, ::Type{T}, d::Int, batch_size::Int) where {T}
    return NumericalShadow.random_unitary(backend, T, d, batch_size)
end

function looped_end_to_end(::Type{T}, d::Int, batch_size::Int) where {T}
    Z = CUDA.randn(T, d, d, batch_size)
    Q = CUDA.zeros(T, d, d, batch_size)

    for i in 1:batch_size
        Q[:, :, i] .= phase_corrected_q_from_qr_slice(view(Z, :, :, i))
    end

    return Q
end

function benchmark_case(label::AbstractString, bench_expr)
    println("Running: $label")
    trial = run(bench_expr)
    return CaseStats(median(trial).time / 1e6, mean(trial).time / 1e6)
end

function print_summary(title::AbstractString, batched::CaseStats, looped::CaseStats)
    median_speedup = looped.median_ms / batched.median_ms
    mean_speedup = looped.mean_ms / batched.mean_ms

    println()
    println(title)
    println(rpad("Case", 30) * lpad("Median (ms)", 14) * lpad("Mean (ms)", 14))
    println("-"^58)
    @printf("%-30s%14.3f%14.3f\n", "Batched", batched.median_ms, batched.mean_ms)
    @printf("%-30s%14.3f%14.3f\n", "Looped single qr", looped.median_ms, looped.mean_ms)
    @printf("%-30s%14.3f%14.3f\n", "Speedup (looped/batched)", median_speedup, mean_speedup)
end

function main()
    ensure_cuda!()
    CUDA.allowscalar(false)

    backend = CUDABackend()
    A_ref = CUDA.randn(T, D, D, BATCH_SIZE)

    # Warmup
    work_warmup = copy(A_ref)
    CUDA.@sync batched_q_from_compact_qr(work_warmup, backend)
    work_warmup = copy(A_ref)
    CUDA.@sync looped_q_from_qr(work_warmup)
    CUDA.@sync batched_end_to_end(backend, T, D, BATCH_SIZE)
    CUDA.@sync looped_end_to_end(T, D, BATCH_SIZE)

    batched_q_from_qr_bench = @benchmarkable begin
        work = copy($A_ref)
        CUDA.@sync batched_q_from_compact_qr(work, $backend)
    end evals = BENCH_EVALS samples = BENCH_SAMPLES seconds = BENCH_SECONDS

    looped_q_from_qr_bench = @benchmarkable begin
        work = copy($A_ref)
        CUDA.@sync looped_q_from_qr(work)
    end evals = BENCH_EVALS samples = BENCH_SAMPLES seconds = BENCH_SECONDS

    batched_e2e_bench = @benchmarkable begin
        CUDA.@sync batched_end_to_end($backend, $T, $D, $BATCH_SIZE)
    end evals = BENCH_EVALS samples = BENCH_SAMPLES seconds = BENCH_SECONDS

    looped_e2e_bench = @benchmarkable begin
        CUDA.@sync looped_end_to_end($T, $D, $BATCH_SIZE)
    end evals = BENCH_EVALS samples = BENCH_SAMPLES seconds = BENCH_SECONDS

    qr_to_q_batched_stats = benchmark_case(
        "QR->Q batched geqrf_batched! + Householder reconstruction",
        batched_q_from_qr_bench,
    )
    qr_to_q_looped_stats = benchmark_case(
        "QR->Q looped qr(view(...)) + phase correction",
        looped_q_from_qr_bench,
    )
    e2e_batched_stats = benchmark_case("End-to-end batched random_unitary", batched_e2e_bench)
    e2e_looped_stats = benchmark_case("End-to-end looped qr baseline", looped_e2e_bench)

    println()
    println("CUDA Batched QR Benchmark")
    @printf("d=%d, batch_size=%d, T=%s\n", D, BATCH_SIZE, string(T))
    @printf("Benchmark config: evals=%d, samples=%d, seconds=%.1f\n", BENCH_EVALS, BENCH_SAMPLES, BENCH_SECONDS)

    print_summary("Fixed input QR->Q path", qr_to_q_batched_stats, qr_to_q_looped_stats)
    print_summary("End-to-end (unitary generation path)", e2e_batched_stats, e2e_looped_stats)
end

main()
