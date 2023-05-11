"""
$(DocStringExtensions.README)
"""
module NeuralPDE

# Fixes #682
using Turing, MacroTools, LinearAlgebra
using Parameters, Distributions, Optim, Requires
using Distances, DocStringExtensions, Random, StanSample
using DynamicHMC, TransformVariables, LogDensityProblemsAD, TransformedLogDensities
STANDARD_PROB_GENERATOR(prob, p) = remake(prob; u0 = eltype(p).(prob.u0), p = p)
function STANDARD_PROB_GENERATOR(prob::EnsembleProblem, p)
    EnsembleProblem(remake(prob.prob; u0 = eltype(p).(prob.prob.u0), p = p))
end

using DocStringExtensions
using Reexport, Statistics
@reexport using DiffEqBase
@reexport using ModelingToolkit

using Zygote, ForwardDiff, Random, Distributions
using Adapt, DiffEqNoiseProcess, StochasticDiffEq
using Optimization
using Integrals, IntegralsCubature
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
import ModelingToolkit: value, nameof, toexpr, build_expr, expand_derivatives
import DomainSets: Domain, ClosedInterval
import ModelingToolkit: Interval, infimum, supremum #,Ball
import SciMLBase: @add_kwonly, parameterless_type
import Optimisers
import UnPack: @unpack
import RecursiveArrayTools
import ChainRulesCore, Flux, Lux, ComponentArrays
import ChainRulesCore: @non_differentiable

RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type AbstractPINN end

abstract type AbstractTrainingStrategy end

# Fixes #682
include("turing_MCMC.jl")

include("pinn_types.jl")
include("symbolic_utilities.jl")
include("training_strategies.jl")
include("adaptive_losses.jl")
include("ode_solve.jl")
include("rode_solve.jl")
include("transform_inf_integral.jl")
include("discretize.jl")
include("neural_adapter.jl")

export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS, NNRODE,
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
       MiniMaxAdaptiveLoss,
       LogOptions

#   Fixes #682
export turing_MCMC

end # module
