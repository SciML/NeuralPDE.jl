@testitem "Aqua" tags = [:qa] begin
    using NeuralPDE, Aqua

    Aqua.test_all(NeuralPDE; ambiguities = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end

@testitem "ExplicitImports" tags = [:qa] begin
    using NeuralPDE, ExplicitImports

    @test check_no_implicit_imports(NeuralPDE) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralPDE) === nothing
end
