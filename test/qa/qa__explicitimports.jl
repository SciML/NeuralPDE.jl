using NeuralPDE
using Test

@testset "ExplicitImports" begin
    using NeuralPDE, ExplicitImports

    # skip ModelingToolkit: NeuralPDE @reexport's it for downstream convenience.
    # ignore ModelingToolkitBase: module reference from using ModelingToolkitBase: @named, ...
    @test check_no_implicit_imports(
        NeuralPDE;
        skip = (Base, Core, ModelingToolkit),
        ignore = (:ModelingToolkitBase,)
    ) === nothing
    @test check_no_stale_explicit_imports(NeuralPDE) === nothing
    # Ignore get_ivs/get_dvs: owned by ModelingToolkitBase but accessed via ModelingToolkit.
    @test check_all_qualified_accesses_via_owners(
        NeuralPDE; ignore = (:get_ivs, :get_dvs)
    ) === nothing
end
