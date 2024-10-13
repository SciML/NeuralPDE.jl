using NeuralPDE, Aqua
@testset "Aqua" begin
    Aqua.test_all(NeuralPDE; ambiguities = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end
