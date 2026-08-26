module NeuralPDE

using ADTypes: ADTypes, AutoForwardDiff, AutoZygote
using Adapt: Adapt
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
using Cubature: Cubature
using ComponentArrays: ComponentArrays, ComponentArray, getdata, getaxes
using ConcreteStructs: @concrete
using DocStringExtensions: FIELDS
using DomainSets: DomainSets, AbstractInterval, leftendpoint, rightendpoint, ProductDomain
using ForwardDiff: ForwardDiff
using Functors: Functors, fmap
using Integrals: Integrals, CubatureJLh, QuadGKJL
using IntervalSets: infimum, supremum
using LinearAlgebra: Diagonal
using Lux: Lux, Chain, Dense, SkipConnection, StatefulLuxLayer
using Lux: FromFluxAdaptor, recursive_eltype
using NeuralOperators: DeepONet
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using MLDataDevices: CPUDevice, get_device
using Optimisers: Optimisers, Adam
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Printf: @printf
using Random: Random, AbstractRNG
using RecursiveArrayTools: DiffEqArray
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
using SciMLBase: SciMLBase, BatchIntegralFunction, DAEProblem, IntegralProblem,
    NoiseProblem, ODEFunction, ODEInputFunction, ODEProblem, ODESolution,
    OptimizationFunction, OptimizationProblem, PDETimeSeriesSolution, ReturnCode,
    SDEProblem, discretize, init, isinplace, remake, solve, symbolic_discretize
using SciMLPublic: @public
using Statistics: Statistics, mean
using QuasiMonteCarlo: QuasiMonteCarlo, LatinHypercubeSample
using WeightInitializers: glorot_uniform, zeros32
using Zygote: Zygote

# Symbolic Stuff
using ModelingToolkit: ModelingToolkit, toexpr
using ModelingToolkitBase: @mtkcompile, @named, @parameters, PDESystem, get_dvs, get_ivs,
    mtkcompile, unknowns
using Symbolics: Symbolics, Differential, Integral, arguments, Num, expand_derivatives,
    @register_symbolic, @variables
using SymbolicUtils: SymbolicUtils, unwrap
using SymbolicIndexingInterface: SymbolicIndexingInterface

# Needed for the Bayesian Stuff
using Distributions: Distributions, Distribution, MvNormal, Normal, dim, logpdf
using MonteCarloMeasurements: Particles

import LuxCore: initialparameters, initialstates, parameterlength

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    AbstractPINN

Abstract supertype for PDE discretizations that use a physics-informed neural
network.

# Fields

This abstract type has no fields. Concrete discretizations define the state
needed by their `symbolic_discretize` method.

# Extension Rules

A concrete subtype must add a method for
`SciMLBase.symbolic_discretize(pde_system::PDESystem, discretization::MyPINN)`.
The method must translate the symbolic `PDESystem` and the discretization into
the symbolic representation consumed by the package's training workflow. Callers
should use the generic `SciMLBase.symbolic_discretize` entry point; the concrete
type is the dispatch extension point.

This is a developer interface. Application code should normally use one of the
concrete discretizations exported by NeuralPDE.

# Example

```julia
using ModelingToolkit: PDESystem

struct MyPINN <: NeuralPDE.AbstractPINN end

function SciMLBase.symbolic_discretize(
        pde_system::PDESystem, discretization::MyPINN
    )
    return (; pde_system, discretization)
end
```
"""
abstract type AbstractPINN end

"""
    AbstractTrainingStrategy

Abstract supertype for the sampling and loss-construction strategies used by
NeuralPDE discretizations.

# Fields

This abstract type has no fields. Concrete strategies define the configuration
needed by their training-data and loss-construction methods.

# Extension Rules

A custom strategy must implement the generic
`get_loss_function(init_params, loss_function, training_data, T, strategy;
kwargs...)` interface. For an interval-based strategy, the training data may
instead be passed as lower and upper bounds:
`get_loss_function(init_params, loss_function, lower_bounds, upper_bounds, T,
strategy; kwargs...)`. In either form, the method must return a callable scalar
objective whose first argument is the optimization parameter container.

`loss_function` receives the strategy's training data and that parameter
container and returns the residuals to aggregate. `T` is the element type used
for generated training data. Keyword arguments are strategy-specific and are
forwarded by the generic training workflow. Implement `generate_training_sets`
or `get_bounds` only when the strategy needs those representations.

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
include("symbolic_utilities.jl")
include("training_strategies.jl")
include("adaptive_losses.jl")

include("ode_solve.jl")
include("dae_solve.jl")
include("pino_ode_solve.jl")
include("transform_inf_integral.jl")
include("discretize.jl")

include("neural_adapter.jl")
include("bpinn_types.jl")

include("dgm.jl")
include("NN_SDE_solve.jl")
include("NN_SDE_weaksolve.jl")

include("precompilation.jl")

export PINOODE
export NNODE, NNDAE
export BNNODE, ahmc_bayesian_pinn_ode, ahmc_bayesian_pinn_pde
export NNSDE
export SDEPINN
export PhysicsInformedNN
export BPINNsolution, BayesianPINN
export DeepGalerkin

export neural_adapter

export GridTraining, StochasticTraining, QuadratureTraining, QuasiRandomTraining,
    WeightedIntervalTraining

export build_loss_function, get_loss_function,
    generate_training_sets, get_variables, get_argument, get_bounds,
    get_numeric_integral, vector_to_parameters

export AbstractAdaptiveLoss, NonAdaptiveLoss, GradientScaleAdaptiveLoss,
    MiniMaxAdaptiveLoss, SoftAdaptAdaptiveLoss, ReLoBRaLoAdaptiveLoss

export LogOptions

export SciMLBase, DAEProblem, NoiseProblem, ODEFunction, ODEInputFunction, ODEProblem,
    ODESolution, OptimizationFunction, OptimizationProblem, PDETimeSeriesSolution,
    ReturnCode, SDEProblem, discretize, init, remake, solve, symbolic_discretize
export ModelingToolkit, Differential, Integral, PDESystem, mtkcompile, unknowns,
    @mtkcompile, @named, @parameters, @register_symbolic, @variables

@public logscalar, logvector

end # module
