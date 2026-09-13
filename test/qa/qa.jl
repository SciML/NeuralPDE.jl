using SciMLTesting, NeuralPDE, Test

# Load every weak dependency so run_qa also scans NeuralPDE's package extensions.
using AdvancedHMC, LogDensityProblems, MCMCChains

# Kept in sync with the reexport `export` blocks in src/NeuralPDE.jl.
const REEXPORTS = (
    # SciML common interface (SciMLBase)
    :SciMLBase, :DAEProblem, :NoiseProblem, :ODEFunction, :ODEInputFunction, :ODEProblem,
    :ODESolution, :OptimizationFunction, :OptimizationProblem, :PDENoTimeSolution,
    :PDETimeSeriesSolution,
    :ReturnCode, :SDEProblem, :discretize, :init, :remake, :solve, :symbolic_discretize,
    # Symbolic front end (ModelingToolkit / ModelingToolkitBase / Symbolics)
    :ModelingToolkit, :Differential, :Integral, :PDESystem, :mtkcompile, :unknowns,
    Symbol("@mtkcompile"), Symbol("@named"), Symbol("@parameters"),
    Symbol("@register_symbolic"), Symbol("@variables"),
)

@testset "Reexported public API stays in scope" begin
    exported = Set(names(NeuralPDE))
    for name in REEXPORTS
        @test name in exported
        @test isdefined(NeuralPDE, name)
    end
end

run_qa(
    NeuralPDE;
    reexports_allow = REEXPORTS,
    ei_kwargs = (;
        # Retain the reviewed Base.mapany, Base.Broadcast.dottable,
        # SymbolicUtils._iszero, and Symbolics.variables calls. ForwardDiff does not
        # declare its derivative entry points public, and AdvancedHMC has no public
        # equivalent for constructing a kernel from its public sampler specifications.
        all_qualified_accesses_are_public = (;
            ignore = (
                :_iszero, :derivative, :dottable, :jacobian, :make_kernel, :mapany,
                :variables,
            ),
        ),
        # ExplicitImports scans extensions as separate modules, although these
        # helpers remain internal to the NeuralPDE package boundary.
        # `tovar` is made public upstream in
        # https://github.com/SciML/ModelingToolkit.jl/pull/5120; drop it here once
        # that is released.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractTrainingStrategy, :BPINNstats, :safe_expand, :safe_get_device,
                :tovar,
            ),
        ),
    ),
)
