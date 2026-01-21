@testitem "Aqua" tags = [:qa] begin
    using NeuralPDE, Aqua

    # Skip undefined_exports: ModelingToolkit exports AbstractDynamicOptProblem but doesn't
    # define it, and this gets re-exported via @reexport (upstream ModelingToolkit issue)
    # Skip persistent_tasks: SymbolicsPreallocationToolsExt has __precompile__(false) which
    # causes Aqua's precompilation check to fail (upstream Symbolics.jl issue)
    Aqua.test_all(NeuralPDE; ambiguities = false, undefined_exports = false,
        persistent_tasks = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end

@testitem "ExplicitImports" tags = [:qa] begin
    using NeuralPDE, ExplicitImports

    @test check_no_implicit_imports(NeuralPDE) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralPDE) === nothing
end
