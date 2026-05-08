module NeuralPDEBPINNExt

using NeuralPDE
using NeuralPDE: NeuralPDE, AbstractTrainingStrategy, GridTraining, StochasticTraining,
    QuadratureTraining, WeightedIntervalTraining, BNNODE, BPINNstats,
    BPINNsolution, vector_to_parameters, build_loss_function,
    get_dataset_train_points, merge_strategy_with_loglikelihood_function,
    safe_get_device, safe_expand

using AdvancedHMC: AdvancedHMC, DiagEuclideanMetric, HMC, HMCDA, Hamiltonian,
    JitteredLeapfrog, Leapfrog, MassMatrixAdaptor, NUTS, StanHMCAdaptor,
    StepSizeAdaptor, TemperedLeapfrog, find_good_stepsize
using ChainRulesCore: @ignore_derivatives
using ComponentArrays: ComponentArrays, ComponentArray, getdata, getaxes
using ConcreteStructs: @concrete
using Distributions: Distributions, Distribution, MvNormal, Normal, dim, logpdf
using ForwardDiff: ForwardDiff
using Functors: fmap
using IntervalSets: infimum, supremum
using LinearAlgebra: Diagonal
using LogDensityProblems: LogDensityProblems
using Lux: Lux, StatefulLuxLayer, FromFluxAdaptor
using LuxCore: LuxCore, AbstractLuxLayer
using MCMCChains: MCMCChains, Chains, sample
using ModelingToolkit: ModelingToolkit, Differential, toexpr
using MonteCarloMeasurements: Particles
using Printf: @printf
using Random: Random
using Integrals: IntegralProblem, QuadGKJL
using SciMLBase: SciMLBase, isinplace, solve, symbolic_discretize
using SymbolicUtils: SymbolicUtils
using Symbolics: Symbolics

include("bpinn/advancedHMC_MCMC.jl")
include("bpinn/BPINN_ode.jl")
include("bpinn/PDE_BPINN.jl")

end # module
