"""
Function needed for converting vector of sampled parameters into ComponentVector in case of Lux chain output, derivatives
the sampled parameters are of exotic type `Dual` due to ForwardDiff's autodiff tagging.
"""
function vector_to_parameters(ps_new::AbstractVector, ps::Union{NamedTuple, ComponentArray})
    @assert length(ps_new) == LuxCore.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

vector_to_parameters(ps_new::AbstractVector, _::AbstractVector) = ps_new

"""
    BNNODE(chain, kernel = AdvancedHMC.HMC; strategy = nothing, draw_samples = 2000,
           priorsNNw = (0.0, 2.0), param = [nothing], l2std = [0.05],
           phystd = [0.05], phynewstd = (ode_params)->[0.05], dataset = [], physdt = 1 / 20.0,
           MCMCargs = (; n_leapfrog=30), nchains = 1, init_params = nothing,
           Adaptorkwargs = (; Adaptor = AdvancedHMC.StanHMCAdaptor, targetacceptancerate = 0.8,
                              Metric = AdvancedHMC.DiagEuclideanMetric),
           Integratorkwargs = (Integrator = AdvancedHMC.Leapfrog,), autodiff = false, estim_collocate = false, progress = false, verbose = false)

Algorithm for solving ordinary differential equations using a Bayesian neural network. This
is a specialization of the physics-informed neural network which is used as a solver for a
standard `ODEProblem`.

!!! warning

    Note that BNNODE only supports ODEs which are written in the out-of-place form, i.e.
    `du = f(u,p,t)`, and not `f(du,u,p,t)`. If not declared out-of-place, then the BNNODE
    will exit with an error.

!!! note

    BNNODE requires `AdvancedHMC`, `MCMCChains`, and `LogDensityProblems` to be loaded
    (e.g. `using AdvancedHMC, MCMCChains, LogDensityProblems`) before it can be used.

## Positional Arguments

* `chain`: A neural network architecture, defined as a `Lux.AbstractLuxLayer`.
* `kernel`: Choice of MCMC Sampling Algorithm. Defaults to `AdvancedHMC.HMC`

## Keyword Arguments

(refer `NeuralPDE.ahmc_bayesian_pinn_ode` keyword arguments.)

## Example

```julia
using NeuralPDE, AdvancedHMC, MCMCChains, LogDensityProblems

linear = (u, p, t) -> -u / p[1] + exp(t / p[2]) * cos(t)
tspan = (0.0, 10.0)
u0 = 0.0
p = [5.0, -5.0]
prob = ODEProblem(linear, u0, tspan, p)
linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))

sol = solve(prob, Tsit5(); saveat = 0.05)
u = sol.u[1:100]
time = sol.t[1:100]
x̂ = u .+ (u .* 0.2) .* randn(size(u))
dataset = [x̂, time, 0.05 .* ones(length(time))]

chainlux = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))

alg = BNNODE(chainlux; draw_samples = 2000, l2std = [0.05], phystd = [0.05],
             priorsNNw = (0.0, 3.0), progress = true)

sol_lux = solve(prob, alg)

# with parameter estimation
alg = BNNODE(chainlux; dataset, draw_samples = 2000, l2std = [0.05], phystd = [0.05],
             priorsNNw = (0.0, 10.0), param = [Normal(6.5, 0.5), Normal(-3, 0.5)],
             progress = true)

sol_lux_pestim = solve(prob, alg)
```

## Solution Notes

Note that the solution is evaluated at fixed time points according to the strategy chosen.
ensemble solution is evaluated and given at steps of `saveat`.
A Dataset must be provided : When doing ODE Parameter Estimation OR if we want to use a Data L2 loss
and/or Data Quadrature loss term (flagged via `estim_collocate`) during a forward solve.
The neural network is a fully continuous solution so `BPINNsolution`
is an accurate interpolation (up to the neural network training result). In addition, the
`BPINNstats` is returned as `sol.fullsolution` for further analysis.

## References

Liu Yanga, Xuhui Menga, George Em Karniadakis. "B-PINNs: Bayesian Physics-Informed Neural
Networks for Forward and Inverse PDE Problems with Noisy Data".

Kevin Linka, Amelie Schäfer, Xuhui Meng, Zongren Zou, George Em Karniadakis, Ellen Kuhl
"Bayesian Physics Informed Neural Networks for real-world nonlinear dynamical systems".
"""
@concrete struct BNNODE <: NeuralPDEAlgorithm
    chain <: AbstractLuxLayer
    kernel
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    draw_samples::Int
    priorsNNw::Tuple{Float64, Float64}
    param <: Union{Nothing, Vector{<:Distribution}}
    l2std::Vector{Float64}
    phystd::Vector{Float64}
    phynewstd
    dataset <: Union{Vector, Vector{<:Vector{<:AbstractFloat}}}
    physdt::Float64
    MCMCkwargs <: NamedTuple
    nchains::Int
    init_params <: Union{Nothing, <:NamedTuple, Vector{<:AbstractFloat}}
    Adaptorkwargs <: NamedTuple
    Integratorkwargs <: NamedTuple
    numensemble::Int
    estim_collocate::Bool
    autodiff::Bool
    progress::Bool
    verbose::Bool
end

"""
Contains `ahmc_bayesian_pinn_ode()` function output:

1. A MCMCChains.jl chain object for sampled parameters.
2. The set of all sampled parameters.
3. Statistics like:
    - n_steps
    - acceptance_rate
    - log_density
    - hamiltonian_energy
    - hamiltonian_energy_error
    - numerical_error
    - step_size
    - nom_step_size
"""
@concrete struct BPINNstats
    mcmc_chain
    samples
    statistics
end

"""
BPINN Solution contains the original solution from AdvancedHMC.jl sampling (BPINNstats
contains fields related to that).

1. `ensemblesol` is the Probabilistic Estimate (MonteCarloMeasurements.jl Particles type) of
   Ensemble solution from All Neural Network's (made using all sampled parameters) output's.
2. `estimated_nn_params` - Probabilistic Estimate of NN params from sampled weights, biases.
3. `estimated_de_params` - Probabilistic Estimate of DE params from sampled unknown DE
   parameters.
"""
@concrete struct BPINNsolution
    original <: BPINNstats
    ensemblesol
    estimated_nn_params
    estimated_de_params
    timepoints
end

"""
    ahmc_bayesian_pinn_ode(prob, chain; kwargs...)

Bayesian inference of an ODE problem via NUTS / HMC sampling. Implemented in the
`NeuralPDEBPINNExt` package extension. Load `AdvancedHMC`, `MCMCChains` and
`LogDensityProblems` to enable it (e.g. `using AdvancedHMC, MCMCChains, LogDensityProblems`).

See the extension method for the full keyword argument documentation.
"""
function ahmc_bayesian_pinn_ode end

"""
    ahmc_bayesian_pinn_pde(pde_system, discretization; kwargs...)

Bayesian inference of a PDE system via NUTS / HMC sampling. Implemented in the
`NeuralPDEBPINNExt` package extension. Load `AdvancedHMC`, `MCMCChains` and
`LogDensityProblems` to enable it (e.g. `using AdvancedHMC, MCMCChains, LogDensityProblems`).

See the extension method for the full keyword argument documentation.
"""
function ahmc_bayesian_pinn_pde end
