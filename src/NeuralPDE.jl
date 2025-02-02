module NeuralPDE

using ADTypes: ADTypes, AutoForwardDiff, AutoZygote
using Adapt: Adapt
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore, @non_differentiable, @ignore_derivatives
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
using Reexport: @reexport
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
using SciMLBase: SciMLBase, BatchIntegralFunction, IntegralProblem, NoiseProblem,
	OptimizationFunction, OptimizationProblem, ReturnCode, discretize,
	isinplace, solve, symbolic_discretize
using Statistics: Statistics, mean
using QuasiMonteCarlo: QuasiMonteCarlo, LatinHypercubeSample
using WeightInitializers: glorot_uniform, zeros32
using Zygote: Zygote

# Symbolic Stuff
using ModelingToolkit: ModelingToolkit, PDESystem, Differential, toexpr
using Symbolics: Symbolics, unwrap, arguments, operation, build_expr, Num,
	expand_derivatives
using SymbolicUtils: SymbolicUtils
using SymbolicIndexingInterface: SymbolicIndexingInterface

# Needed for the Bayesian Stuff
using AdvancedHMC: AdvancedHMC, HMCDA,
	NUTS
using Distributions: Distributions, Distribution, MvNormal, dim, logpdf
using LogDensityProblems: LogDensityProblems
using MCMCChains: MCMCChains

import LuxCore: initialparameters, initialstates, parameterlength

@reexport using SciMLBase, ModelingToolkit

RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type AbstractPINN end

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
include("advancedHMC_MCMC.jl")

include("dgm.jl")

export PINOODE
export NNODE, NNDAE
export PhysicsInformedNN, discretize
export DeepGalerkin

export neural_adapter

export GridTraining, StochasticTraining, QuadratureTraining, QuasiRandomTraining,
	WeightedIntervalTraining

export build_loss_function, get_loss_function,
	generate_training_sets, get_variables, get_argument, get_bounds,
	get_numeric_integral, symbolic_discretize, vector_to_parameters

export AbstractAdaptiveLoss, NonAdaptiveLoss, GradientScaleAdaptiveLoss,
	MiniMaxAdaptiveLoss

export LogOptions

end # module
