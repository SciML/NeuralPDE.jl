module NeuralPDEBPINNExt

using NeuralPDE
using NeuralPDE: NeuralPDE, AbstractTrainingStrategy, GridTraining, StochasticTraining,
    QuadratureTraining, WeightedIntervalTraining, BNNODE, BPINNstats,
    BPINNsolution, BayesianPINN, PhysicsInformedNN, vector_to_parameters,
    safe_get_device, safe_expand, unique_networks, pinn_metadata, TrialNetwork

using AdvancedHMC: AdvancedHMC, DiagEuclideanMetric, HMC, HMCDA, Hamiltonian,
    JitteredLeapfrog, Leapfrog, MassMatrixAdaptor, NUTS, StanHMCAdaptor,
    StepSizeAdaptor, TemperedLeapfrog, find_good_stepsize
using ADTypes: AutoForwardDiff
using ComponentArrays: ComponentArrays, ComponentArray, getdata
using ConcreteStructs: @concrete
using Distributions: Distributions, Distribution, MvNormal, logpdf
using DomainSets: DomainSets
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
using SciMLBase: SciMLBase, isinplace, solve, remake
using SymbolicIndexingInterface: getu
using SymbolicUtils: SymbolicUtils
using Symbolics: Symbolics, unwrap, iscall, operation, arguments
using ModelingToolkitBase: get_ivs, get_domain, getdefault
using ModelingToolkit: ModelingToolkit

include("bpinn/advancedHMC_MCMC.jl")
include("bpinn/BPINN_ode.jl")
include("bpinn/PDE_BPINN.jl")

end # module
