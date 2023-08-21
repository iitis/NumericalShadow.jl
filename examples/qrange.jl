using NumericalShadow
using LinearAlgebra
using CUDA
using ProgressMeter

samples = 10^8
batchsize = 10^8
T = ComplexF32
d = 2
U = Array(qr(rand(ComplexF32, d^2, d^2)).Q)
U = Array(Diagonal([1, exp(1im * π/3), exp(1im * 2π/3), exp(1im * 3π/3)]))
@showprogress 2 "Iteratring q" for q=0.01:0.01:1
   shadow = NumericalShadow.qshadow_GPU(U, samples, q, batchsize)
   shadow.nr = NumericalShadow.numerical_range(U)
   shadow.evs = eigvals(U)
   shadow.other_range = NumericalShadow.qrange(U, q)
   NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/qshadow_$(rpad(q, 4, "0")).npz",
    )
end

@showprogress 2 "Iteratring q" for q=0.01:0.01:1
    shadow = NumericalShadow.product_qshadow_GPU(U, samples, q, batchsize)
    shadow.nr = NumericalShadow.numerical_range(U)
    shadow.evs = eigvals(U)
    shadow.other_range = NumericalShadow.qrange(U, q)
    NumericalShadow.save(
         shadow,
         "$(@__DIR__)/results/product_qshadow_$(rpad(q, 4, "0")).npz",
     )
 end