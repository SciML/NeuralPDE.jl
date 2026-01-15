@testitem "Aqua" tags = [:qa] begin
    using NeuralPDE, Aqua

    # Skip undefined exports check for Variable and find_solvables! which are
    # re-exported from ModelingToolkit but no longer exist in MTK 11
    Aqua.test_all(NeuralPDE; ambiguities = false, undefined_exports = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end

@testitem "ExplicitImports" tags = [:qa] begin
    using NeuralPDE, ExplicitImports

    @test check_no_implicit_imports(NeuralPDE) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralPDE) === nothing
end
