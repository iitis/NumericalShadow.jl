using Aqua
using Test

@testset "Aqua" begin
    Aqua.test_all(NumericalShadow; ambiguities = false)
end
