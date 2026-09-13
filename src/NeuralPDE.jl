module NeuralPDE

using ADTypes: ADTypes, AutoForwardDiff, AutoZygote
using Adapt: Adapt
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
using Cubature: Cubature
using ComponentArrays: ComponentArrays, ComponentArray
using ConcreteStructs: @concrete
using DomainSets: DomainSets
using Enzyme: Enzyme
using FastGaussQuadrature: gausslegendre
using ForwardDiff: ForwardDiff
using Functors: Functors, fmap
using Integrals: Integrals, CubatureJLh, GaussLegendre, QuadGKJL
using LinearAlgebra: I
using Lux: Lux, Chain, Dense, SkipConnection, StatefulLuxLayer
using Lux: FromFluxAdaptor, recursive_eltype
using NeuralOperators: DeepONet
using LuxCore: LuxCore, AbstractLuxContainerLayer, AbstractLuxLayer, AbstractLuxWrapperLayer
using MLDataDevices: CPUDevice, get_device
using ModelingToolkitNeuralNets: ModelingToolkitNeuralNets, SymbolicNeuralNetwork
using Optimisers: Optimisers, Adam
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using PDEBase: PDEBase
using Printf: @printf
using Random: Random, AbstractRNG
using RecursiveArrayTools: DiffEqArray
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions
using SciMLBase: SciMLBase, BatchIntegralFunction, DAEProblem, IntegralProblem,
    NoiseProblem, ODEFunction, ODEInputFunction, ODEProblem, ODESolution,
    OptimizationFunction, OptimizationProblem, PDENoTimeSolution, PDETimeSeriesSolution,
    ReturnCode, SDEProblem, discretize, init, isinplace, remake, solve,
    symbolic_discretize
using SciMLPublic: @public
using Statistics: Statistics, mean
using QuasiMonteCarlo: QuasiMonteCarlo, LatinHypercubeSample
using WeightInitializers: glorot_uniform, zeros32
using Zygote: Zygote

# Symbolic Stuff
using ModelingToolkit: ModelingToolkit
using ModelingToolkitBase: ModelingToolkitBase, @mtkcompile, @named, @parameters, complete,
    PDESystem, ProblemTypeCtx, System, get_bcs, get_dvs, get_ivs, get_ps,
    getdefault, initial_conditions, mtkcompile, setdefault, tovar, unknowns
using Symbolics: Symbolics, Differential, Equation, Integral, arguments, iscall, Num, operation,
    wrap, @register_symbolic, @variables
using SymbolicUtils: SymbolicUtils, getmetadata, unwrap
using SymbolicIndexingInterface: SymbolicIndexingInterface, getu, setp

# Needed for the Bayesian Stuff
using Distributions: Distributions, Distribution, Normal
using MonteCarloMeasurements: Particles

import LuxCore: initialparameters, parameterlength

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    AbstractTrainingStrategy

Abstract supertype for the sampling and loss-construction strategies used by
NeuralPDE discretizations and solvers.

# Fields

This abstract type has no fields. Concrete strategies define the configuration
needed by their training-data and loss-construction methods.

# Extension Rules

A strategy used with [`PhysicsInformedNN`](@ref) implements the collocation interface
`collocation_count(strategy, kind, ivpos, bounds, pinned)` and
`sample_points(strategy, block, rng)`, optionally `resamples(strategy)` and
`uses_quadrature_weights(strategy)`.

A strategy used with the ODE solvers implements the generic
`get_loss_function(init_params, loss_function, training_data, T, strategy;
kwargs...)` interface. For an interval-based strategy, the training data may
instead be passed as lower and upper bounds:
`get_loss_function(init_params, loss_function, lower_bounds, upper_bounds, T,
strategy; kwargs...)`. In either form, the method must return a callable scalar
objective whose first argument is the optimization parameter container.

This is a developer interface. User code should generally use the built-in
training strategies.

# Example

```julia
struct MyTraining <: NeuralPDE.AbstractTrainingStrategy end

function NeuralPDE.get_loss_function(
        init_params, loss_function, training_data, T, ::MyTraining; scale = 1
    )
    return θ -> scale * sum(abs2, loss_function(training_data, θ))
end
```
"""
abstract type AbstractTrainingStrategy end

const cdev = CPUDevice()

@inline safe_get_device(x) = safe_get_device(get_device(x), x)
@inline safe_get_device(::Nothing, x) = cdev
@inline safe_get_device(dev, _) = dev

@inline safe_expand(dev, x) = dev(x)
@inline safe_expand(::CPUDevice, x::AbstractRange) = x
@inline safe_collect(dev, x::AbstractRange) = dev(collect(x))

include("eltype_matching.jl")

include("pinn_types.jl")
include("training_strategies.jl")
include("pinn_lowering.jl")
include("discretize.jl")
include("pde_solution.jl")

include("ode_solve.jl")
include("dae_solve.jl")
include("pino_ode_solve.jl")

include("bpinn_types.jl")

include("dgm.jl")
include("NN_SDE_solve.jl")
include("NN_SDE_weaksolve.jl")

include("precompilation.jl")

export PINOODE
export NNODE, NNDAE
export BNNODE, ahmc_bayesian_pinn_ode
export NNSDE
export SDEPINN
export PhysicsInformedNN, FiniteDifferenceDerivative
export BPINNsolution
export DeepGalerkin

export GridTraining, StochasticTraining, QuadratureTraining, QuasiRandomTraining,
    WeightedIntervalTraining

export get_loss_function, vector_to_parameters
export pinn_metadata, resample!

export SciMLBase, DAEProblem, NoiseProblem, ODEFunction, ODEInputFunction, ODEProblem,
    ODESolution, OptimizationFunction, OptimizationProblem, PDENoTimeSolution,
    PDETimeSeriesSolution, ReturnCode, SDEProblem, discretize, init, remake, solve,
    symbolic_discretize
export ModelingToolkit, Differential, Integral, PDESystem, mtkcompile, unknowns,
    @mtkcompile, @named, @parameters, @register_symbolic, @variables

@public AbstractDerivativeLowering, PINNMetadata, ResidualBlock, TrialNetwork,
    AdditionalLoss, nn_eval, nn_eval_row, default_adtype, lower, lower_derivative,
    trial_function,
    collocation_count, sample_points, resamples, uses_quadrature_weights

end # module
