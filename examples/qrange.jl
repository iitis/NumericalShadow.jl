using NumericalShadow
using LinearAlgebra
using CUDA
using ProgressMeter

function LinearAlgebra.kron!(z::CuMatrix, x::CuMatrix, y::CuMatrix)
    function kernel(z, x, y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i <= size(z, 2)
            d1 = size(x, 1)
            d2 = size(y, 1)
            for k=1:d1, l=1:d2
                z[(k-1)*d2+l, i] = x[k, i] * y[l, i]
            end
        end
        return
    end
    @assert size(z, 2) == size(x, 2) == size(y, 2)
    @assert size(z, 1) == size(x, 1) * size(y, 1)
    threads = 512
    blocks = cld(size(x, 2), threads)
    @cuda threads=threads blocks=blocks kernel(z, x, y)
end

samples = 10^8
batchsize = 10^8
T = ComplexF32
d = 2
U = Array(qr(rand(ComplexF32, d^2, d^2)).Q)
@showprogress 2 "Iteratring q" for q=0.01:0.01:1
   shadow = NumericalShadow.qshadow_GPU(U, samples, q, batchsize)
   shadow.nr = NumericalShadow.numerical_range(U)
   shadow.evs = eigvals(U)
   NumericalShadow.save(
        shadow,
        "$(@__DIR__)/results/qrange_$(rpad(q, 4, "0")).npz",
    )
end
