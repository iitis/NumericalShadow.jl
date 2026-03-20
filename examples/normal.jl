# Shadow.pdf mapping:
#   Problem 1 baseline (d = 0 normal matrix case).
# Use this to compare with the non-normal sweep in `hidden_non_normal.jl`.

using NumericalShadow
using KernelAbstractions
using LinearAlgebra
using CUDA

function choose_backend()
    use_cuda = lowercase(get(ENV, "NUMERICALSHADOW_USE_CUDA", "true")) in ("1", "true", "yes")
    if use_cuda && CUDA.functional()
        println("Using CUDA backend.")
        return CUDABackend()
    end
    println("Using CPU backend.")
    return CPU()
end

backend = choose_backend()
samples = parse(Int, get(ENV, "NUMERICALSHADOW_SAMPLES", "100000000"))
batchsize = parse(Int, get(ENV, "NUMERICALSHADOW_BATCHSIZE", "1000000"))
batchsize > 0 || throw(ArgumentError("`batchsize` must be positive, got $batchsize"))
samples > 0 || throw(ArgumentError("`samples` must be positive, got $samples"))

A = collect(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]))
sampling_f = (b, d, n) -> NumericalShadow.random_pure(b, ComplexF32, d, n)
shadow = NumericalShadow.shadow(backend, A, samples, sampling_f, batchsize)
shadow.nr = NumericalShadow.numerical_range(A)
shadow.evs = eigvals(A)
NumericalShadow.save(
    shadow,
    "$(@__DIR__)/results/normal_complex.h5",
)

A = collect(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]))
sampling_f = (b, d, n) -> NumericalShadow.random_pure(b, Float32, d, n)
shadow = NumericalShadow.shadow(backend, A, samples, sampling_f, batchsize)
shadow.nr = NumericalShadow.numerical_range(A)
shadow.evs = eigvals(A)
NumericalShadow.save(
    shadow,
    "$(@__DIR__)/results/normal_real.h5",
)
