module NeuralPDEBPINNExt

using NeuralPDE
using NeuralPDE: NeuralPDE, AbstractTrainingStrategy, GridTraining, StochasticTraining,
    QuadratureTraining, WeightedIntervalTraining, BNNODE, BPINNstats,
    BPINNsolution, vector_to_parameters, safe_get_device, safe_expand

using AdvancedHMC: AdvancedHMC, DiagEuclideanMetric, HMC, HMCDA, Hamiltonian,
    JitteredLeapfrog, Leapfrog, MassMatrixAdaptor, NUTS, StanHMCAdaptor,
    StepSizeAdaptor, TemperedLeapfrog, find_good_stepsize
using ComponentArrays: ComponentArrays, ComponentArray, getdata
using ConcreteStructs: @concrete
using Distributions: Distributions, Distribution, MvNormal, logpdf
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal
using LogDensityProblems: LogDensityProblems
using Lux: Lux, StatefulLuxLayer, FromFluxAdaptor
using LuxCore: LuxCore, AbstractLuxLayer
using MCMCChains: MCMCChains, Chains, sample
using MonteCarloMeasurements: Particles
using Printf: @printf
using Random: Random
using Integrals: IntegralProblem, QuadGKJL
using SciMLBase: SciMLBase, isinplace, solve
using SymbolicUtils: SymbolicUtils
using Symbolics: Symbolics

include("bpinn/advancedHMC_MCMC.jl")
include("bpinn/BPINN_ode.jl")

end # module
