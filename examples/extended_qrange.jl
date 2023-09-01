using NumericalShadow
using LinearAlgebra
using CUDA
using ProgressMeter

function swap(n, perm)
    @assert length(perm) == n
    U = zeros(2^n, 2^n)
    for i=0:2^n-1
        jj = reverse(reverse(digits(i; base=2, pad=n))[perm])
        j = sum(2^(k-1)*l for (k, l) in enumerate(jj))
        U[i+1, j+1] = 1
    end
    return U
end

samples = 10^10
batchsize = 10^7
T = ComplexF32
n = 4
S = swap(n, [3, 1, 2, 4])
U1 = S*kron(Array(Diagonal([1, exp(1im * π/3), exp(1im * 2π/3), exp(1im * 3π/3)])), I(4))*S'
U = Array(Diagonal([1, exp(1im * π/3), exp(1im * 2π/3), exp(1im * 3π/3)]))
U = kron(kron(I(2), U), I(2))

# @showprogress 2 "Iteratring q" for q=0.01:0.01:1
#    shadow = NumericalShadow.qshadow_GPU(T, U, samples, q, batchsize)
#    shadow.nr = NumericalShadow.numerical_range(U)
#    shadow.evs = eigvals(U)
#    shadow.other_range = NumericalShadow.qrange(U, q)
#    NumericalShadow.save(
#         shadow,
#         "$(@__DIR__)/results/extended_qshadow_complex_$(rpad(q, 4, "0")).npz",
#     )
# end

# @showprogress 2 "Iteratring q" for q=0.01:0.01:1
#     shadow = NumericalShadow.product_qshadow_GPU(T, U, samples, q, batchsize)
#     shadow.nr = NumericalShadow.numerical_range(U)
#     shadow.evs = eigvals(U)
#     shadow.other_range = NumericalShadow.qrange(U, q)
#     NumericalShadow.save(
#          shadow,
#          "$(@__DIR__)/results/extended_product_qshadow_complex_$(rpad(q, 4, "0")).npz",
#      )
#  end

T = Float32
@showprogress 2 "Iteratring q" for q=0.01:0.01:1
   shadow = NumericalShadow.qshadow_GPU(T, U, samples, q, batchsize)
   shadow.nr = NumericalShadow.numerical_range(U)
   shadow.evs = eigvals(U)
   shadow.other_range = NumericalShadow.qrange(U, q)
   NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/extended_qshadow_real_$(rpad(q, 4, "0")).npz",
    )
end

@showprogress 2 "Iteratring q" for q=0.01:0.01:1
    shadow = NumericalShadow.product_qshadow_GPU(T, U, samples, q, batchsize)
    shadow.nr = NumericalShadow.numerical_range(U)
    shadow.evs = eigvals(U)
    shadow.other_range = NumericalShadow.qrange(U, q)
    NumericalShadow.save(
         shadow,
         "$(@__DIR__)/results/extended_product_qshadow_real_$(rpad(q, 4, "0")).npz",
     )
 end