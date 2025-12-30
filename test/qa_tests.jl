@testitem "Aqua" tags=[:qa] begin
    using NeuralPDE, Aqua

    Aqua.test_all(NeuralPDE; ambiguities = false)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
end

@testitem "ExplicitImports" tags=[:qa] begin
    using NeuralPDE, ExplicitImports

    @test check_no_implicit_imports(NeuralPDE) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralPDE) === nothing
end

@testitem "JET" tags=[:qa] begin
    using NeuralPDE, JET

    # Test the package using JET's test_package function
    # This runs static analysis on the package and reports any issues found
    # We use mode=:typo to focus on detecting potential errors like undefined references
    # Full analysis with mode=:basic can be expensive for large packages
    JET.test_package(NeuralPDE; target_defined_modules = true, mode = :typo)
end
