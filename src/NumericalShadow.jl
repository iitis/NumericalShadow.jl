module NumericalShadow
using LinearAlgebra
using CUDA
using NPZ

const nTPB = 256

include("random_matrices.jl")
include("range.jl")
include("shadow.jl")
include("histogram.jl")

end
