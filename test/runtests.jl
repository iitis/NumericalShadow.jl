using NumericalShadow
using Test

# Try to load CUDA for testing.
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset verbose = true "NumericalShadow.jl" begin
    include("test_aqua.jl")
    include("test_random_helpers.jl")
    include("test_shadow_workflow.jl")
    include("test_ka_qr_cpu.jl")
end
