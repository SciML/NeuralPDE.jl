using NeuralPDE, Aqua, ExplicitImports

@testset "Aqua" begin
    Aqua.test_all(NeuralPDE; ambiguities = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(NeuralPDE) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralPDE) === nothing
end
