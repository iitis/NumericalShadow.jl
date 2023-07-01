using NumericalShadow
using LinearAlgebra
using CUDA

samples = 10^9
batchsize = 10^8
T = ComplexF32
for d = 0:0.01:0.5
    A = cat(Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]), [0 d; 0 0], dims = (1, 2))
    sampling_f = (x...) -> NumericalShadow.random_pure(T, x...)
    shadow = NumericalShadow.shadow_GPU(A, samples, sampling_f, batchsize)
    shadow.nr = NumericalShadow.numerical_range(A)
    NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/hidden_non_normal_$(rpad(d, 4, "0")).npz",
    )
end
