@testitem "Aqua" tags = [:qa] begin
    using NeuralPDE, Aqua

    # Skip undefined_exports check because ModelingToolkit has a broken export
    # for AbstractDynamicOptProblem that gets re-exported via @reexport.
    # This is an upstream issue in ModelingToolkit, not NeuralPDE.
    Aqua.test_all(NeuralPDE; ambiguities = false, undefined_exports = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end

@testitem "ExplicitImports" tags = [:qa] begin
    using NeuralPDE, ExplicitImports

    @test check_no_implicit_imports(NeuralPDE) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralPDE) === nothing
end
