module NumericalShadow
using LinearAlgebra
using KernelAbstractions
using NPZ

include("random_matrices.jl")
include("range.jl")
include("shadow.jl")
include("histogram.jl")

function move_to_backend(backend, data)
    data
end

export move_to_backend, shadow, qshadow, product_qshadow

end
