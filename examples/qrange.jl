using NumericalShadow
using KernelAbstractions
using LinearAlgebra
using CUDA
using ProgressMeter

CUDA.functional() || error("CUDA is not functional. This example requires a working CUDA setup.")
backend = CUDABackend()

samples = 10^8
batchsize = 10^8
T = ComplexF32
d = 2
U = Array(qr(rand(ComplexF32, d^2, d^2)).Q)
U = Array(Diagonal([1, exp(1im * π/3), exp(1im * 2π/3), exp(1im * 3π/3)]))
@showprogress 2 "Iteratring q" for q=0.01:0.01:1
   shadow = NumericalShadow.qshadow(backend, T, U, samples, q, batchsize)
   shadow.nr = NumericalShadow.numerical_range(U)
   shadow.evs = eigvals(U)
   shadow.other_range = NumericalShadow.qrange(U, q)
   NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/qshadow_complex_$(rpad(q, 4, "0")).npz",
    )
end

@showprogress 2 "Iteratring q" for q=0.01:0.01:1
    shadow = NumericalShadow.product_qshadow(backend, T, U, samples, q, batchsize)
    shadow.nr = NumericalShadow.numerical_range(U)
    shadow.evs = eigvals(U)
    shadow.other_range = NumericalShadow.qrange(U, q)
    NumericalShadow.save(
         shadow,
         "$(@__DIR__)/results/product_qshadow_complex_$(rpad(q, 4, "0")).npz",
     )
 end

samples = 10^8
batchsize = 10^8
T = Float32
d = 2
U = Array(qr(rand(ComplexF32, d^2, d^2)).Q)
U = Array(Diagonal([1, exp(1im * π/3), exp(1im * 2π/3), exp(1im * 3π/3)]))
@showprogress 2 "Iteratring q" for q=0.01:0.01:1
   shadow = NumericalShadow.qshadow(backend, T, U, samples, q, batchsize)
   shadow.nr = NumericalShadow.numerical_range(U)
   shadow.evs = eigvals(U)
   shadow.other_range = NumericalShadow.qrange(U, q)
   NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/qshadow_real_$(rpad(q, 4, "0")).npz",
    )
end

@showprogress 2 "Iteratring q" for q=0.01:0.01:1
    shadow = NumericalShadow.product_qshadow(backend, T, U, samples, q, batchsize)
    shadow.nr = NumericalShadow.numerical_range(U)
    shadow.evs = eigvals(U)
    shadow.other_range = NumericalShadow.qrange(U, q)
    NumericalShadow.save(
         shadow,
         "$(@__DIR__)/results/product_qshadow_real_$(rpad(q, 4, "0")).npz",
     )
 end
