module NumericalShadow
using LinearAlgebra
using CUDA
using NPZ

const nTPB = 256

function my_kron!(z::CuMatrix, x::CuMatrix, y::CuMatrix)
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

include("random_matrices.jl")
include("range.jl")
include("shadow.jl")
include("histogram.jl")

end
