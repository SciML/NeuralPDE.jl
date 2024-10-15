"""
$(DocStringExtensions.README)
"""
module NeuralPDE

using ADTypes: ADTypes, AutoForwardDiff, AutoZygote
using Adapt: Adapt
using AdvancedHMC: AdvancedHMC, DiagEuclideanMetric, HMC, HMCDA, Hamiltonian,
                   JitteredLeapfrog, Leapfrog, MassMatrixAdaptor, NUTS, StanHMCAdaptor,
                   StepSizeAdaptor, TemperedLeapfrog, find_good_stepsize
using ArrayInterface: ArrayInterface, parameterless_type
using ChainRulesCore: ChainRulesCore, @non_differentiable, @ignore_derivatives
using Cubature: Cubature
using ComponentArrays: ComponentArrays, ComponentArray, getdata, getaxes
using ConcreteStructs: @concrete
using Distributions: Distributions, Distribution, MvNormal, Normal, dim, logpdf
using DocStringExtensions: DocStringExtensions, FIELDS
using DomainSets: DomainSets, AbstractInterval, leftendpoint, rightendpoint, ProductDomain
using ForwardDiff: ForwardDiff
using Functors: Functors, fmap
using Integrals: Integrals, CubatureJLh, QuadGKJL
using LinearAlgebra: Diagonal
using LogDensityProblems: LogDensityProblems
using Lux: Lux, Chain, Dense, SkipConnection, StatefulLuxLayer
using Lux: FromFluxAdaptor, recursive_eltype
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using MCMCChains: MCMCChains, Chains, sample
using ModelingToolkit: ModelingToolkit, Num, PDESystem, toexpr, expand_derivatives, infimum,
                       supremum
using MonteCarloMeasurements: Particles
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
using Symbolics: Symbolics, unwrap, arguments, operation, build_expr
using SymbolicUtils: SymbolicUtils
using SymbolicIndexingInterface: SymbolicIndexingInterface
using QuasiMonteCarlo: QuasiMonteCarlo, LatinHypercubeSample
using WeightInitializers: glorot_uniform, zeros32
using Zygote: Zygote

import LuxCore: initialparameters, initialstates, parameterlength

@reexport using SciMLBase, ModelingToolkit

RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type AbstractPINN end

abstract type AbstractTrainingStrategy end

include("eltype_matching.jl")

include("pinn_types.jl")
include("symbolic_utilities.jl")
include("training_strategies.jl")
include("adaptive_losses.jl")

include("ode_solve.jl")
include("rode_solve.jl")
include("dae_solve.jl")

include("transform_inf_integral.jl")
include("discretize.jl")

include("neural_adapter.jl")
include("advancedHMC_MCMC.jl")
include("BPINN_ode.jl")
include("PDE_BPINN.jl")

include("dgm.jl")

export NNODE, NNDAE, NNRODE
export BNNODE, ahmc_bayesian_pinn_ode, ahmc_bayesian_pinn_pde
export PhysicsInformedNN, discretize
export BPINNsolution, BayesianPINN
export DeepGalerkin

export GridTraining, StochasticTraining, QuadratureTraining, QuasiRandomTraining,
       WeightedIntervalTraining

export build_loss_function, get_loss_function,
       generate_training_sets, get_variables, get_argument, get_bounds,
       get_numeric_integral, symbolic_discretize, vector_to_parameters

export AbstractAdaptiveLoss, NonAdaptiveLoss, GradientScaleAdaptiveLoss,
       MiniMaxAdaptiveLoss

export LogOptions

end # module
