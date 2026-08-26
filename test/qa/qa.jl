using SciMLTesting, NeuralPDE, Test

# Load every weak dependency so run_qa also scans NeuralPDE's package extensions.
using AdvancedHMC, LogDensityProblems, MCMCChains, TensorBoardLogger

# Kept in sync with the reexport `export` blocks in src/NeuralPDE.jl.
const REEXPORTS = (
    # SciML common interface (SciMLBase)
    :SciMLBase, :DAEProblem, :NoiseProblem, :ODEFunction, :ODEInputFunction, :ODEProblem,
    :ODESolution, :OptimizationFunction, :OptimizationProblem, :PDETimeSeriesSolution,
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
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractTrainingStrategy, :BPINNstats, :get_dataset_train_points,
                :merge_strategy_with_loglikelihood_function, :safe_expand,
                :safe_get_device,
            ),
        ),
    ),
)

mutable struct RecordingLogger
    scalars::Dict{Tuple{String, Int}, Real}
    vectors::Dict{Tuple{String, Int}, Vector{Float64}}
end

RecordingLogger() = RecordingLogger(
    Dict{Tuple{String, Int}, Real}(), Dict{Tuple{String, Int}, Vector{Float64}}()
)

function NeuralPDE.logscalar(
        logger::RecordingLogger, value::Real, name::AbstractString, step::Integer
    )
    logger.scalars[(name, step)] = value
    return nothing
end

function NeuralPDE.logvector(
        logger::RecordingLogger, values::AbstractVector{<:Real}, name::AbstractString,
        step::Integer
    )
    logger.vectors[(name, step)] = Float64.(values)
    return nothing
end

@testset "Public logging hooks" begin
    logger = RecordingLogger()
    @test NeuralPDE.logscalar(logger, 1.5, "loss", 2) === nothing
    @test NeuralPDE.logvector(logger, [1.0, 2.0], "weights", 3) === nothing
    @test logger.scalars[("loss", 2)] == 1.5
    @test logger.vectors[("weights", 3)] == [1.0, 2.0]
end
