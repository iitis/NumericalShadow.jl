"""
    NumericalShadow

Numerical shadows, q-shadows, and random-state/matrix generators with
CPU and optional CUDA backends.
"""
module NumericalShadow
using LinearAlgebra
using KernelAbstractions
using HDF5

include("random_matrices.jl")
include("range.jl")
include("shadow.jl")
include("histogram.jl")

"""
    move_to_backend(backend, data)

Move `data` to the selected `backend`.

The default implementation is a no-op for CPU-like backends.
Extensions can provide backend-specific transfers (for example, CUDA).
"""
function move_to_backend(backend, data)
    data
end

export move_to_backend
export shadow, qshadow, product_qshadow
export random_pure, random_overlap, random_unitary, random_max_ent

end
