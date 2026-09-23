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

# Extensions triggered only by weakdeps are unschedulable by parallel
# precompilation and can outlive the precompile driver; they must stay opted
# out (#1203).
@testset "Weakdep extensions opt out of precompilation" begin
    proj = Base.parsed_toml(joinpath(pkgdir(NeuralPDE), "Project.toml"))
    weakdeps = Set(keys(get(proj, "weakdeps", Dict{String,Any}())))
    for (ext, triggers) in get(proj, "extensions", Dict{String,Any}())
        issubset(Set(triggers), weakdeps) || continue
        extfile = joinpath(pkgdir(NeuralPDE), "ext", ext * ".jl")
        @test occursin(r"__precompile__\(\s*false\s*\)", read(extfile, String))
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
