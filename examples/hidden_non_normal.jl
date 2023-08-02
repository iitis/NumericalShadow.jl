using NumericalShadow
using LinearAlgebra
using CUDA
using ProgressMeter

samples = 10^10
batchsize = 10^8
T = ComplexF32
step = 0.01
@showprogress 2 "Iteratring d" for d = 0:step:1.0
    A = cat(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]), [0 d; 0 0], dims = (1, 2))
    sampling_f = (x...) -> NumericalShadow.random_pure(T, x...)
    shadow = NumericalShadow.shadow_GPU(A, samples, sampling_f, batchsize)
    shadow.nr = NumericalShadow.numerical_range(A)
    shadow.evs = eigvals(A)
    NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/hidden_non_normal_complex_$(rpad(d, 4, "0")).npz",
    )
end

T = Float32
for d = 0:step:1.0
    A = cat(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]), [0 d; 0 0], dims = (1, 2))
    sampling_f = (x...) -> NumericalShadow.random_pure(T, x...)
    shadow = NumericalShadow.shadow_GPU(A, samples, sampling_f, batchsize)
    shadow.nr = NumericalShadow.numerical_range(A)
    shadow.evs = eigvals(A)
    NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/hidden_non_normal_real_$(rpad(d, 4, "0")).npz",
    )
end
