"""
    ahmc_bayesian_pinn_ode(prob, chain; strategy = GridTraining, dataset = [nothing],
                           init_params = nothing, draw_samples = 1000, physdt = 1 / 20.0f0,
                           l2std = [0.05], phystd = [0.05], phynewstd = [0.05], priorsNNw = (0.0, 2.0),
                           param = [], nchains = 1, autodiff = false, Kernel = HMC,
                           Adaptorkwargs = (Adaptor = StanHMCAdaptor,
                               Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
                           Integratorkwargs = (Integrator = Leapfrog,),
                           MCMCkwargs = (n_leapfrog = 30,), progress = false,
                           verbose = false)

!!! warn

    Note that `ahmc_bayesian_pinn_ode()` only supports ODEs which are written in the
    out-of-place form, i.e. `du = f(u,p,t)`, and not `f(du,u,p,t)`. If not declared
    out-of-place, then `ahmc_bayesian_pinn_ode()` will exit with an error.

## Example

```julia
linear = (u, p, t) -> -u / p[1] + exp(t / p[2]) * cos(t)
tspan = (0.0, 10.0)
u0 = 0.0
p = [5.0, -5.0]
prob = ODEProblem(linear, u0, tspan, p)

### CREATE DATASET (Necessity for accurate Parameter estimation)
sol = solve(prob, Tsit5(); saveat = 0.05)
u = sol.u[1:100]
time = sol.t[1:100]

### dataset and BPINN create
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
dataset = [x̂, time]

chain1 = Lux.Chain(Lux.Dense(1, 5, tanh), Lux.Dense(5, 5, tanh), Lux.Dense(5, 1)

### simply solving ode here hence better to not pass dataset(uses ode params specified in prob)
fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chain1,
                                                            dataset = dataset,
                                                            draw_samples = 1500,
                                                            l2std = [0.05],
                                                            phystd = [0.05],
                                                            priorsNNw = (0.0,3.0))

### solving ode + estimating parameters hence dataset needed to optimize parameters upon + Pior Distributions for ODE params
fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chain1,
                                                            dataset = dataset,
                                                            draw_samples = 1500,
                                                            l2std = [0.05],
                                                            phystd = [0.05],
                                                            priorsNNw = (0.0,3.0),
                                                            param = [Normal(6.5,0.5), Normal(-3,0.5)])
```

## NOTES

Dataset is required for accurate Parameter estimation + solving equations
Incase you are only solving the Equations for solution, do not provide dataset

## Positional Arguments

* `prob`: DEProblem(out of place and the function signature should be f(u,p,t).
* `chain`: Lux Neural Netork which would be made the Bayesian PINN.

## Keyword Arguments

* `strategy`: The training strategy used to choose the points for the evaluations. By
  default GridTraining is used with given physdt discretization.
* `init_params`: initial parameter values for BPINN (ideally for multiple chains different
  initializations preferred)
* `nchains`: number of chains you want to sample
* `draw_samples`: number of samples to be drawn in the MCMC algorithms (warmup samples are
  ~2/3 of draw samples)
* `l2std`: standard deviation of BPINN prediction against L2 losses/Dataset
* `phystd`: standard deviation of BPINN prediction against Chosen Underlying ODE System
* `phynewstd`: standard deviation of new loss func term
* `priorsNNw`: Tuple of (mean, std) for BPINN Network parameters. Weights and Biases of
  BPINN are Normal Distributions by default.
* `param`: Vector of chosen ODE parameters Distributions in case of Inverse problems.
* `autodiff`: Boolean Value for choice of Derivative Backend(default is numerical)
* `physdt`: Timestep for approximating ODE in it's Time domain. (1/20.0 by default)
* `Kernel`: Choice of MCMC Sampling Algorithm (AdvancedHMC.jl implementations HMC/NUTS/HMCDA)
* `Integratorkwargs`: `Integrator`, `jitter_rate`, `tempering_rate`.
  Refer: https://turinglang.org/AdvancedHMC.jl/stable/
* `Adaptorkwargs`: `Adaptor`, `Metric`, `targetacceptancerate`.
  Refer: https://turinglang.org/AdvancedHMC.jl/stable/ Note: Target percentage (in decimal)
  of iterations in which the proposals are accepted (0.8 by default)
* `MCMCargs`: A NamedTuple containing all the chosen MCMC kernel's (HMC/NUTS/HMCDA)
  Arguments, as follows :
    * `n_leapfrog`: number of leapfrog steps for HMC
    * `δ`: target acceptance probability for NUTS and HMCDA
    * `λ`: target trajectory length for HMCDA
    * `max_depth`: Maximum doubling tree depth (NUTS)
    * `Δ_max`: Maximum divergence during doubling tree (NUTS)
    Refer: https://turinglang.org/AdvancedHMC.jl/stable/
* `progress`: controls whether to show the progress meter or not.
* `verbose`: controls the verbosity. (Sample call args in AHMC)

!!! warning

    AdvancedHMC.jl is still developing convenience structs so might need changes on new
    releases.
"""
function ahmc_bayesian_pinn_ode(
        prob::SciMLBase.ODEProblem, chain; strategy = GridTraining, dataset = [nothing],
        init_params = nothing, draw_samples = 1000, physdt = 1 / 20.0, l2std = [0.05],
        phystd = [0.05], phynewstd = [0.05], priorsNNw = (0.0, 2.0), param = [], nchains = 1,
        autodiff = false, Kernel = HMC,
        Adaptorkwargs = (Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
        Integratorkwargs = (Integrator = Leapfrog,), MCMCkwargs = (n_leapfrog = 30,),
        progress = false, verbose = false, estim_collocate = false)
    @assert !isinplace(prob) "The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."

    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))

    strategy = strategy == GridTraining ? strategy(physdt) : strategy

    if dataset != [nothing] &&
       (length(dataset) < 2 || !(dataset isa Vector{<:Vector{<:AbstractFloat}}))
        error("Invalid dataset. dataset would be timeseries (x̂,t) where type: Vector{Vector{AbstractFloat}")
    end

    if dataset != [nothing] && param == []
        println("Dataset is only needed for Parameter Estimation + Forward Problem, not in only Forward Problem case.")
    elseif dataset == [nothing] && param != []
        error("Dataset Required for Parameter Estimation.")
    end

    initial_nnθ, chain, st = generate_ltd(chain, init_params)

    @assert nchains≤Threads.nthreads() "number of chains is greater than available threads"
    @assert nchains≥1 "number of chains must be greater than 1"

    # eltype(physdt) cause needs Float64 for find_good_stepsize
    # Lux chain(using component array later as vector_to_parameter need namedtuple)
    T = eltype(physdt)
    initial_θ = getdata(ComponentArray{T}(initial_nnθ))

    # adding ode parameter estimation
    nparameters = length(initial_θ)
    ninv = length(param)
    priors = [
        MvNormal(T(priorsNNw[1]) * ones(T, nparameters),
        Diagonal(abs2.(T(priorsNNw[2]) .* ones(T, nparameters))))
    ]

    # append Ode params to all paramvector
    if ninv > 0
        # shift ode params(initialise ode params by prior means)
        initial_θ = vcat(initial_θ, [Distributions.params(param[i])[1] for i in 1:ninv])
        priors = vcat(priors, param)
        nparameters += ninv
    end

    smodel = StatefulLuxLayer{true}(chain, nothing, st)
    # dimensions would be total no of params,initial_nnθ for Lux namedTuples
    ℓπ = LogTargetDensity(nparameters, prob, smodel, strategy, dataset, priors,
        phystd, phynewstd, l2std, autodiff, physdt, ninv, initial_nnθ, estim_collocate)

    if verbose
        @printf("Current Physics Log-likelihood: %g\n", physloglikelihood(ℓπ, initial_θ))
        @printf("Current Prior Log-likelihood: %g\n", priorweights(ℓπ, initial_θ))
        @printf("Current SSE against dataset Log-likelihood: %g\n",
            L2LossData(ℓπ, initial_θ))
        if estim_collocate
            @printf("Current gradient loss against dataset Log-likelihood: %g\n",
                L2loss2(ℓπ, initial_θ))
        end
    end

    Adaptor = Adaptorkwargs[:Adaptor]
    Metric = Adaptorkwargs[:Metric]
    targetacceptancerate = Adaptorkwargs[:targetacceptancerate]

    # Define Hamiltonian system (nparameters ~ dimensionality of the sampling space)
    metric = Metric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    # parallel sampling option
    if nchains != 1
        # Cache to store the chains
        chains = Vector{Any}(undef, nchains)
        statsc = Vector{Any}(undef, nchains)
        samplesc = Vector{Any}(undef, nchains)

        Threads.@threads for i in 1:nchains
            # each chain has different initial NNparameter values(better posterior exploration)
            initial_θ = vcat(
                randn(eltype(initial_θ), nparameters - ninv),
                initial_θ[(nparameters - ninv + 1):end]
            )
            initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
            integrator = integratorchoice(Integratorkwargs, initial_ϵ)
            adaptor = adaptorchoice(Adaptor, MassMatrixAdaptor(metric),
                StepSizeAdaptor(targetacceptancerate, integrator))

            MCMC_alg = kernelchoice(Kernel, MCMCkwargs)
            Kernel = AdvancedHMC.make_kernel(MCMC_alg, integrator)
            samples, stats = sample(hamiltonian, Kernel, initial_θ, draw_samples, adaptor;
                progress = progress, verbose = verbose)

            samplesc[i] = samples
            statsc[i] = stats
            mcmc_chain = Chains(reduce(hcat, samples)')
            chains[i] = mcmc_chain
        end

        return chains, samplesc, statsc
    else
        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = integratorchoice(Integratorkwargs, initial_ϵ)
        adaptor = adaptorchoice(Adaptor, MassMatrixAdaptor(metric),
            StepSizeAdaptor(targetacceptancerate, integrator))

        MCMC_alg = kernelchoice(Kernel, MCMCkwargs)
        Kernel = AdvancedHMC.make_kernel(MCMC_alg, integrator)
        samples, stats = sample(hamiltonian, Kernel, initial_θ, draw_samples,
            adaptor; progress = progress, verbose = verbose)

        if verbose
            println("Sampling Complete.")
            @printf("Final Physics Log-likelihood: %g\n",
                physloglikelihood(ℓπ, samples[end]))
            @printf("Final Prior Log-likelihood: %g\n", priorweights(ℓπ, samples[end]))
            @printf("Final SSE against dataset Log-likelihood: %g\n",
                L2LossData(ℓπ, samples[end]))
            if estim_collocate
                @printf("Final gradient loss against dataset Log-likelihood: %g\n",
                    L2loss2(ℓπ, samples[end]))
            end
        end

        # return a chain(basic chain),samples and stats
        matrix_samples = reshape(hcat(samples...), (length(samples[1]), length(samples), 1))
        mcmc_chain = MCMCChains.Chains(matrix_samples)
        return mcmc_chain, samples, stats
    end
end