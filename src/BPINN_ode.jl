# HIGH level API for BPINN ODE solver

"""
    BNNODE(chain, kernel = HMC; strategy = nothing, draw_samples = 2000,
           priorsNNw = (0.0, 2.0), param = [nothing], l2std = [0.05],
           phystd = [0.05], phynewstd = (ode_params)->[0.05], dataset = [], physdt = 1 / 20.0,
           MCMCargs = (; n_leapfrog=30), nchains = 1, init_params = nothing,
           Adaptorkwargs = (; Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
                              Metric = DiagEuclideanMetric),
           Integratorkwargs = (Integrator = Leapfrog,), autodiff = false, estim_collocate = false, progress = false, verbose = false)

Algorithm for solving ordinary differential equations using a Bayesian neural network. This
is a specialization of the physics-informed neural network which is used as a solver for a
standard `ODEProblem`.

!!! warn

    Note that BNNODE only supports ODEs which are written in the out-of-place form, i.e.
    `du = f(u,p,t)`, and not `f(du,u,p,t)`. If not declared out-of-place, then the BNNODE
    will exit with an error.

## Positional Arguments

* `chain`: A neural network architecture, defined as a `Lux.AbstractLuxLayer`.
* `kernel`: Choice of MCMC Sampling Algorithm. Defaults to `AdvancedHMC.HMC`

## Keyword Arguments

(refer `NeuralPDE.ahmc_bayesian_pinn_ode` keyword arguments.)

## Example

```julia
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
Dataset should only be provided when ODE parameter Estimation is being done.
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

function BNNODE(chain, kernel = HMC; strategy = nothing, draw_samples = 1000,
        priorsNNw = (0.0, 2.0), param = nothing, l2std = [0.05], phystd = [0.05],
        phynewstd = (ode_params) -> [0.05], dataset = [], physdt = 1 / 20.0,
        MCMCkwargs = (n_leapfrog = 30,), nchains = 1, init_params = nothing,
        Adaptorkwargs = (Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
        Integratorkwargs = (Integrator = Leapfrog,),
        numensemble = floor(Int, draw_samples / 3),
        estim_collocate = false, autodiff = false, progress = false, verbose = false)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return BNNODE(chain, kernel, strategy, draw_samples, priorsNNw, param, l2std, phystd,
        phynewstd, dataset, physdt, MCMCkwargs, nchains, init_params, Adaptorkwargs,
        Integratorkwargs, numensemble, estim_collocate, autodiff, progress, verbose)
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

function SciMLBase.__solve(prob::SciMLBase.ODEProblem, alg::BNNODE, args...; dt = nothing,
        timeseries_errors = true, save_everystep = true, adaptive = false,
        abstol = 1.0f-6, reltol = 1.0f-3, verbose = false, saveat = 1 / 50.0,
        maxiters = nothing)
    (; chain, param, strategy, draw_samples, numensemble, verbose) = alg

    # ahmc_bayesian_pinn_ode needs param=[] for easier vcat operation for full vector of parameters
    param = param === nothing ? [] : param
    strategy = strategy === nothing ? GridTraining : strategy

    @assert alg.draw_samples≥0 "Number of samples to be drawn has to be >=0."

    mcmcchain, samples,
    statistics = ahmc_bayesian_pinn_ode(
        prob, chain; strategy, alg.dataset, alg.draw_samples, alg.init_params,
        alg.physdt, alg.l2std, alg.phystd, alg.phynewstd,
        alg.priorsNNw, param, alg.nchains, alg.autodiff,
        Kernel = alg.kernel, alg.Adaptorkwargs, alg.Integratorkwargs,
        alg.MCMCkwargs, alg.progress, alg.verbose, alg.estim_collocate)

    fullsolution = BPINNstats(mcmcchain, samples, statistics)
    ninv = length(param)
    t = collect(eltype(saveat), prob.tspan[1]:saveat:prob.tspan[2])

    θinit, st = LuxCore.setup(Random.default_rng(), chain)
    θ = [vector_to_parameters(samples[i][1:(end - ninv)], θinit)
         for i in (draw_samples - numensemble):draw_samples]

    luxar = [chain(t', θ[i], st)[1] for i in 1:numensemble]
    # only need for size
    θinit = collect(ComponentArray(θinit))

    # constructing ensemble predictions
    ensemblecurves = Vector{}[]
    # check if NN output is more than 1
    numoutput = size(luxar[1])[1]
    if numoutput > 1
        # Initialize a vector to store the separated outputs for each output dimension
        output_matrices = [Vector{Vector{Float32}}() for _ in 1:numoutput]

        # Loop through each element in `luxar`
        for element in luxar
            for i in 1:numoutput
                push!(output_matrices[i], element[i, :])  # Append the i-th output (i-th row) to the i-th output_matrices
            end
        end

        for r in 1:numoutput
            ensem_r = hcat(output_matrices[r]...)'
            ensemblecurve_r = prob.u0[r] .+
                              [Particles(ensem_r[:, i]) for i in 1:length(t)] .*
                              (t .- prob.tspan[1])
            push!(ensemblecurves, ensemblecurve_r)
        end

    else
        ensemblecurve = prob.u0 .+
                        [Particles(reduce(vcat, luxar)[:, i]) for i in 1:length(t)] .*
                        (t .- prob.tspan[1])
        push!(ensemblecurves, ensemblecurve)
    end

    nnparams = length(θinit)
    estimnnparams = [Particles(reduce(hcat, samples[(end - numensemble):end])[i, :])
                     for i in 1:nnparams]

    if ninv == 0
        estimated_params = [nothing]
    else
        estimated_params = [Particles(reduce(hcat, samples[(end - numensemble):end])[i, :])
                            for i in (nnparams + 1):(nnparams + ninv)]
    end

    return BPINNsolution(fullsolution, ensemblecurves, estimnnparams, estimated_params, t)
end
