"""
$(DocStringExtensions.README)
"""
module NeuralPDE

using DocStringExtensions
using Reexport, Statistics
@reexport using DiffEqBase
@reexport using ModelingToolkit

using Zygote, ForwardDiff, Random, Distributions
using Adapt, DiffEqNoiseProcess, StochasticDiffEq
using Optimization
using OptimizationOptimisers
using Integrals, Cubature
using QuasiMonteCarlo
using RuntimeGeneratedFunctions
using SciMLBase
using Statistics
using ArrayInterface
import Optim
using DomainSets
using Symbolics
using Symbolics: wrap, unwrap, arguments, operation
using SymbolicUtils
using AdvancedHMC, LogDensityProblems, LinearAlgebra, Functors, MCMCChains
using MonteCarloMeasurements

import ModelingToolkit: value, nameof, toexpr, build_expr, expand_derivatives
import DomainSets: Domain, ClosedInterval
import ModelingToolkit: Interval, infimum, supremum #,Ball
import SciMLBase: @add_kwonly, parameterless_type
import UnPack: @unpack
import ChainRulesCore, Lux, ComponentArrays
import ChainRulesCore: @non_differentiable

RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type AbstractPINN end

abstract type AbstractTrainingStrategy end

include("pinn_types.jl")
include("symbolic_utilities.jl")
include("training_strategies.jl")
include("adaptive_losses.jl")
include("ode_solve.jl")
# include("rode_solve.jl")
include("dae_solve.jl")
include("transform_inf_integral.jl")
include("discretize.jl")
include("neural_adapter.jl")
include("advancedHMC_MCMC.jl")
include("BPINN_ode.jl")
include("PDE_BPINN.jl")

export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS, NNRODE, NNDAE,
    KolmogorovPDEProblem, NNKolmogorov, NNStopping, ParamKolmogorovPDEProblem,
    KolmogorovParamDomain, NNParamKolmogorov,
    PhysicsInformedNN, discretize,
    GridTraining, StochasticTraining, QuadratureTraining, QuasiRandomTraining,
    WeightedIntervalTraining,
    build_loss_function, get_loss_function,
    generate_training_sets, get_variables, get_argument, get_bounds,
    get_phi, get_numeric_derivative, get_numeric_integral,
    build_symbolic_equation, build_symbolic_loss_function, symbolic_discretize,
    AbstractAdaptiveLoss, NonAdaptiveLoss, GradientScaleAdaptiveLoss,
    MiniMaxAdaptiveLoss, LogOptions,
    ahmc_bayesian_pinn_ode, BNNODE, ahmc_bayesian_pinn_pde, vector_to_parameters,
    BPINNsolution, BayesianPINN

end # module
