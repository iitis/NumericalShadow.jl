using NumericalShadow
using LinearAlgebra
using CUDA
using ProgressMeter
using QuantumInformation
using Random

Random.seed!(42)
samples = 10^10
batchsize = 10^7
T = ComplexF32
d = 9
local_d = isqrt(d)
U = rand(CUE(local_d))
V = rand(CUE(local_d))
UV = kron(U, V)
# W = [exp(2π * im * j * k / d) for j=0:d-1, k=0:d-1] / sqrt(d)
W = I(d) - 2 / local_d * proj(vec(I(local_d)))

@showprogress 2 "Iteratring α" offset=1 for α=0.0:0.01:1
    A = (UV)^(1-α) * W^α
    shadow = shadow_GPU(T, A, samples, random_product, batchsize)
    shadow.nr = NumericalShadow.numerical_range(A)
    shadow.evs = eigvals(A)
    NumericalShadow.save(
         shadow,
         "$(@__DIR__)/results/product_shadow_reflection_d=$(d)_$(rpad(α, 4, "0")).npz";
     )
 end