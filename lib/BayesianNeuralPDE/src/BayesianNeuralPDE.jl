module BayesianNeuralPDE

using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
	AdvancedHMC, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements
using Printf: @printf
using ConcreteStructs: @concrete
using NeuralPDE: PhysicsInformedNN
using SciMLBase: SciMLBase
using ChainRulesCore: ChainRulesCore, @non_differentiable, @ignore_derivatives
using LogDensityProblems: LogDensityProblems

abstract type AbstractPINN end

abstract type AbstractTrainingStrategy end
abstract type NeuralPDEAlgorithm <: SciMLBase.AbstractODEAlgorithm end

include("advancedHMC_MCMC.jl")
include("pinn_types.jl")
include("BPINN_ode.jl")
include("discretize.jl")
include("PDE_BPINN.jl")

export BNNODE, ahmc_bayesian_pinn_ode, ahmc_bayesian_pinn_pde
export BPINNsolution, BayesianPINN

end