@concrete struct LogTargetDensity
    dim::Int
    prob <: SciMLBase.ODEProblem
    smodel <: StatefulLuxLayer
    strategy <: AbstractTrainingStrategy
    dataset <: Union{Vector{Nothing}, Vector{<:Vector{<:AbstractFloat}}}
    priors <: Vector{<:Distribution}
    phystd::Vector{Float64}
    phynewstd::Function
    l2std::Vector{Float64}
    autodiff::Bool
    physdt::Float64
    extraparams::Int
    init_params <: Union{NamedTuple, ComponentArray}
    estim_collocate::Bool
end

"""
NN OUTPUT AT t,θ ~ phi(t,θ).
"""
function (f::LogTargetDensity)(t::AbstractVector, θ)
    θ = vector_to_parameters(θ, f.init_params)
    dev = safe_get_device(θ)
    t = safe_expand(dev, t)
    u0 = f.prob.u0 |> dev
    return u0 .+ (t' .- f.prob.tspan[1]) .* f.smodel(t', θ)
end

(f::LogTargetDensity)(t::Number, θ) = f([t], θ)[:, 1]

"""
Similar to ode_dfdx() in NNODE.
"""
function ode_dfdx(phi::LogTargetDensity, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        return ForwardDiff.jacobian(Base.Fix2(phi, θ), t)
    else
        ϵ = sqrt(eps(eltype(t)))
        return (phi(t .+ ϵ, θ) .- phi(t, θ)) ./ ϵ
    end
end

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

function LogDensityProblems.logdensity(ltd::LogTargetDensity, θ)
    ldensity = physloglikelihood(ltd, θ) + priorweights(ltd, θ) + L2LossData(ltd, θ)
    ltd.estim_collocate && return ldensity + L2loss2(ltd, θ)
    return ldensity
end

LogDensityProblems.dimension(ltd::LogTargetDensity) = ltd.dim

function LogDensityProblems.capabilities(::LogTargetDensity)
    return LogDensityProblems.LogDensityOrder{1}()
end

"""
suggested extra loss function for ODE solver case
"""
@views function L2loss2(ltd::LogTargetDensity, θ)
    ltd.extraparams ≤ 0 && return false  # XXX: type-stability?
    u0 = ltd.prob.u0
    f = ltd.prob.f
    t = ltd.dataset[end - 1]
    û = ltd.dataset[1:(end - 2)]
    quadrature_weights = ltd.dataset[end]

    nnsol = ode_dfdx(ltd, t, θ[1:(length(θ) - ltd.extraparams)], ltd.autodiff)

    ode_params = ltd.extraparams == 1 ? θ[((length(θ) - ltd.extraparams) + 1)] :
                 θ[((length(θ) - ltd.extraparams) + 1):length(θ)]
    phynewstd = ltd.phynewstd(ode_params)

    physsol = if length(u0) == 1
        [f(û[1][i], ode_params, tᵢ) for (i, tᵢ) in enumerate(t)]
    else
        [f([û[j][i] for j in eachindex(u0)], ode_params, tᵢ)
         for (i, tᵢ) in enumerate(t)]
    end
    # form of NN output matrix output dim x n
    deri_physsol = reduce(hcat, physsol)
    T = promote_type(eltype(deri_physsol), eltype(nnsol))

    physlogprob = T(0)
    # for BPINNS Quadrature is NOT applied on timewise logpdfs, it isnt being driven to zero.
    # Gridtraining/trapezoidal rule quadrature_weights is dt.*ones(T, length(t))
    # dims of phynewstd is same as u0 due to BNNODE being an out-of-place ODE solver.
    for i in eachindex(u0)
        physlogprob += logpdf(
            MvNormal((nnsol[i, :] .- deri_physsol[i, :]) .* quadrature_weights,
                Diagonal(abs2.(T(phynewstd[i]) .* ones(T, length(t))))),
            zeros(length(t))
        )
    end
    return physlogprob
end

"""
L2 loss loglikelihood(needed for ODE parameter estimation).
"""
@views function L2LossData(ltd::LogTargetDensity, θ)
    (ltd.dataset isa Vector{Nothing} || ltd.extraparams == 0) && return 0

    # matrix(each row corresponds to vector u's rows)
    nn = ltd(ltd.dataset[end - 1], θ[1:(length(θ) - ltd.extraparams)])
    T = eltype(nn)

    L2logprob = zero(T)
    for i in 1:length(ltd.prob.u0)
        # for u[i] ith vector must be added to dataset,nn[1, :] is the dx in lotka_volterra
        L2logprob += logpdf(
            MvNormal(
                nn[i, :],
                Diagonal(abs2.(T(ltd.l2std[i]) .* ones(T, length(ltd.dataset[i]))))
            ),
            ltd.dataset[i]
        )
    end
    return L2logprob
end

"""
Physics loglikelihood over problem timespan + dataset timepoints.
"""
function physloglikelihood(ltd::LogTargetDensity, θ)
    (; f, p, tspan) = ltd.prob
    (; autodiff, strategy) = ltd

    # parameter estimation chosen or not
    if ltd.extraparams > 0
        ode_params = ltd.extraparams == 1 ? θ[((length(θ) - ltd.extraparams) + 1)] :
                     θ[((length(θ) - ltd.extraparams) + 1):length(θ)]
    else
        ode_params = p isa SciMLBase.NullParameters ? Float64[] : p
    end

    return getlogpdf(strategy, ltd, f, autodiff, tspan, ode_params, θ)
end

function getlogpdf(strategy::GridTraining, ltd::LogTargetDensity, f, autodiff::Bool,
        tspan, ode_params, θ)
    ts = collect(eltype(strategy.dx), tspan[1]:(strategy.dx):tspan[2])
    t = ltd.dataset isa Vector{Nothing} ? ts : vcat(ts, ltd.dataset[end - 1])
    return sum(innerdiff(ltd, f, autodiff, t, θ, ode_params))
end

function getlogpdf(strategy::StochasticTraining, ltd::LogTargetDensity,
        f, autodiff::Bool, tspan, ode_params, θ)
    T = promote_type(eltype(tspan[1]), eltype(tspan[2]))
    samples = (tspan[2] - tspan[1]) .* rand(T, strategy.points) .+ tspan[1]
    t = ltd.dataset isa Vector{Nothing} ? samples : vcat(samples, ltd.dataset[end - 1])
    return sum(innerdiff(ltd, f, autodiff, t, θ, ode_params))
end

function getlogpdf(strategy::QuadratureTraining, ltd::LogTargetDensity, f, autodiff::Bool,
        tspan, ode_params, θ)
    # integrand is shape of NN output
    integrand(t::Number, θ) = innerdiff(ltd, f, autodiff, [t], θ, ode_params)
    intprob = IntegralProblem(
        integrand, (tspan[1], tspan[2]), θ; nout = length(ltd.prob.u0))
    sol = solve(intprob, QuadGKJL(); strategy.abstol, strategy.reltol)
    # sum over losses for all NN outputs
    return sum(sol.u)
end

function getlogpdf(strategy::WeightedIntervalTraining, ltd::LogTargetDensity, f,
        autodiff::Bool, tspan, ode_params, θ)
    minT, maxT = tspan
    weights = strategy.weights ./ sum(strategy.weights)
    N = length(weights)
    difference = (maxT - minT) / N

    ts = eltype(difference)[]
    for (index, item) in enumerate(weights)
        temp_data = rand(1, trunc(Int, strategy.points * item)) .* difference .+ minT .+
                    ((index - 1) * difference)
        append!(ts, temp_data)
    end

    t = ltd.dataset isa Vector{Nothing} ? ts : vcat(ts, ltd.dataset[end - 1])
    return sum(innerdiff(ltd, f, autodiff, t, θ, ode_params))
end

"""
MvNormal likelihood at each `ti` in time `t` for ODE collocation residue with NN with parameters θ.
"""
@views function innerdiff(ltd::LogTargetDensity, f, autodiff::Bool, t::AbstractVector, θ,
        ode_params)
    # ltd used for phi and LogTargetDensity object attributes access
    out = ltd(t, θ[1:(length(θ) - ltd.extraparams)])

    # reject samples case(write clear reason why)
    (any(isinf, out[:, 1]) || any(isinf, ode_params)) && return convert(eltype(out), -Inf)

    # this is a vector{vector{dx,dy}}(handle case single u(float passed))
    if length(out[:, 1]) == 1
        physsol = [f(out[:, i][1], ode_params, t[i]) for i in eachindex(t)]
    else
        physsol = [f(out[:, i], ode_params, t[i]) for i in eachindex(t)]
    end
    physsol = reduce(hcat, physsol)

    nnsol = ode_dfdx(ltd, t, θ[1:(length(θ) - ltd.extraparams)], autodiff)
    T = eltype(nnsol)

    # N dimensional vector if N outputs for NN(each row has logpdf of u[i] where u is vector
    # of dependant variables)
    return [logpdf(
                MvNormal((nnsol[i, :] .- physsol[i, :]),
                    Diagonal(abs2.(T(ltd.phystd[i]) .* ones(T, length(t))))),
                zeros(T, length(t))
            ) for i in 1:length(ltd.prob.u0)]
end

"""
Prior logpdf for NN parameters + ODE constants.
"""
@views function priorweights(ltd::LogTargetDensity, θ)
    allparams = ltd.priors
    nnwparams = allparams[1] # nn weights

    ltd.extraparams ≤ 0 && return logpdf(nnwparams, θ)

    # Vector of ode parameters priors
    invpriors = allparams[2:end]

    invlogpdf = sum(
        logpdf(invpriors[length(θ) - i + 1], θ[i])
    for i in (length(θ) - ltd.extraparams + 1):length(θ))

    return invlogpdf + logpdf(nnwparams, θ[1:(length(θ) - ltd.extraparams)])
end

function generate_ltd(chain::AbstractLuxLayer, init_params)
    return init_params, chain, LuxCore.initialstates(Random.default_rng(), chain)
end

function generate_ltd(chain::AbstractLuxLayer, ::Nothing)
    θ, st = LuxCore.setup(Random.default_rng(), chain)
    return θ, chain, st
end

function kernelchoice(Kernel, MCMCkwargs)
    if Kernel == HMCDA
        Kernel(MCMCkwargs[:δ], MCMCkwargs[:λ])
    elseif Kernel == NUTS
        δ, max_depth, Δ_max = MCMCkwargs[:δ], MCMCkwargs[:max_depth], MCMCkwargs[:Δ_max]
        Kernel(δ; max_depth, Δ_max)
    else # HMC
        Kernel(MCMCkwargs[:n_leapfrog])
    end
end

"""
    ahmc_bayesian_pinn_ode(prob, chain; strategy = GridTraining, dataset = [nothing],
                           init_params = nothing, draw_samples = 1000, physdt = 1 / 20.0f0,
                           l2std = [0.05], phystd = [0.05], phynewstd = (ode_params)->[0.05], priorsNNw = (0.0, 2.0),
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
* `phynewstd`: Function in ode_params that gives the standard deviation of the new loss function terms.
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
        phystd = [0.05], phynewstd = (ode_params) -> [0.05],
        priorsNNw = (0.0, 2.0), param = [], nchains = 1,
        autodiff = false, Kernel = HMC,
        Adaptorkwargs = (Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
        Integratorkwargs = (Integrator = Leapfrog,), MCMCkwargs = (n_leapfrog = 30,),
        progress = false, verbose = false, estim_collocate = false)
    @assert !isinplace(prob) "The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."

    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))

    strategy = strategy == GridTraining ? strategy(physdt) : strategy

    if dataset != [nothing] &&
       (length(dataset) < 3 || !(dataset isa Vector{<:Vector{<:AbstractFloat}}))
        error("Invalid dataset. dataset would be timeseries (x̂,t,W) where type: Vector{Vector{AbstractFloat}")
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
            samples,
            stats = sample(hamiltonian, Kernel, initial_θ, draw_samples, adaptor;
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
        samples,
        stats = sample(hamiltonian, Kernel, initial_θ, draw_samples,
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
