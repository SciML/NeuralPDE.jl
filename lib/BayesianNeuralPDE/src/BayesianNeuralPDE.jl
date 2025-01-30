module BayesianNeuralPDE

using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
      AdvancedHMC, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements

include("advancedHMC_MCMC.jl")
include("BPINN_ode.jl")
include("discretize.jl")
include("PDE_BPINN.jl")
include("pinn_types.jl")

export BNNODE, ahmc_bayesian_pinn_ode, ahmc_bayesian_pinn_pde
export BPINNsolution, BayesianPINN

end