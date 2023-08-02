using NumericalShadow
using LinearAlgebra

samples = 10^10
batchsize = 10^8
T = ComplexF32

A = collect(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]))
sampling_f = (x...) -> NumericalShadow.random_pure(T, x...)
shadow = NumericalShadow.shadow_GPU(A, samples, sampling_f, batchsize)
shadow.nr = NumericalShadow.numerical_range(A)
shadow.evs = eigvals(A)
NumericalShadow.save(
    shadow,
    "$(@__DIR__)/results/normal_complex.npz",
)

T = Float32
A = collect(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]))
sampling_f = (x...) -> NumericalShadow.random_pure(T, x...)
shadow = NumericalShadow.shadow_GPU(A, samples, sampling_f, batchsize)
shadow.nr = NumericalShadow.numerical_range(A)
shadow.evs = eigvals(A)
NumericalShadow.save(
    shadow,
    "$(@__DIR__)/results/normal_real.npz",
)