"""
```julia
ahmc_bayesian_pinn_ode(prob, chain;
                       dataset = [[]],init_params = nothing, nchains = 1,
                       draw_samples = 1000, l2std = [0.05],
                       phystd = [0.05], priorsNNw = (0.0, 2.0), param = [],
                       autodiff = false, physdt = 1 / 20.0f0,
                       Proposal = StaticTrajectory,
                       Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
                       Integrator = Leapfrog, Metric = DiagEuclideanMetric)
```

## Example
linear = (u, p, t) -> -u / p[1] + exp(t / p[2]) * cos(t)
tspan = (0.0, 10.0)
u0 = 0.0
p = [5.0, -5.0]
prob = ODEProblem(linear, u0, tspan, p)

# CREATE DATASET (Necessity for accurate Parameter estimation)
sol = solve(prob, Tsit5(); saveat = 0.05)
u = sol.u[1:100]
time = sol.t[1:100]

# dataset and BPINN create
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
dataset = [x̂, time]

chainflux1 = Flux.Chain(Flux.Dense(1, 5, tanh), Flux.Dense(5, 5, tanh), Flux.Dense(5, 1)

# simply solving ode here hence better to not pass dataset(uses ode params specified in prob)
fh_mcmc_chainflux1, fhsamplesflux1, fhstatsflux1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                                          draw_samples = 1000,
                                                                          l2std = [0.05],
                                                                          phystd = [0.05],
                                                                          priorsNNw = (0.0,
                                                                                       3.0))
# solving ode + estimating parameters hence dataset needed to optimize parameters upon
fh_mcmc_chainflux2, fhsamplesflux2, fhstatsflux2 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                                          dataset = dataset,
                                                                          draw_samples = 1000,
                                                                          l2std = [0.05],
                                                                          phystd = [0.05],
                                                                          priorsNNw = (0.0,
                                                                                       3.0),
                                                                          param = [
                                                                              Normal(6.5,
                                                                                     2),
                                                                              Normal(-3, 2),
                                                                          ])
## Positional Arguments
prob -> DEProblem(out of place and the function signature should be f(u,p,t)
chain -> Lux/Flux Neural Netork which would be made the Bayesian PINN
dataset -> Vector containing Vectors of corresponding u,t values 
init_params -> intial parameter values for BPINN (ideally for multiple chains different initializations preferred)
nchains -> number of chains you want to sample (random initialisation of params by default)
draw_samples -> number of samples to be drawn in the MCMC algorithms (warmup samples are ~2/3 of draw samples)
l2std -> standard deviation of BPINN predicition against L2 losses/Dataset
phystd -> standard deviation of BPINN predicition against Chosen Underlying ODE System
priorsNNw -> Vector of [mean, std] for BPINN parameter. Weights and Biases of BPINN are Normal Distributions by default
param -> Vector of chosen ODE parameters Distributions in case of Inverse problems.
autodiff -> Boolean Value for choice of Derivative Backend(default is numerical)
physdt -> Timestep for approximating ODE in it's Time domain. (1/20.0 by default)

#update as AdvancedHMC has added named structs for algos
Proposal -> Choice of MCMC Sampling Algorithm (AdvancedHMC.jl implemenations)
targetacceptancerate -> Target percentage(in decimal) of iterations in which the proposals were accepted(0.8 by default)
Adaptor -> https://turinglang.org/AdvancedHMC.jl/stable/
Integrator -> https://turinglang.org/AdvancedHMC.jl/stable/
Metric -> https://turinglang.org/AdvancedHMC.jl/stable/

## References  

"""
mutable struct PDELogTargetDensity{C, S, I, P <: Vector{Distribution}}
    autodiff::Bool
    extraparams::Int

    prob::Any
    dim::Int
    priors::P
    dataset::Vector{Vector{Float64}}
    l2std::Vector{Float64}
    phystd::Vector{Float64}
    pde_losses::Any
    bc_losses::Any
    phi::Any

    function PDELogTargetDensity(dim, prob, chain::Optimisers.Restructure, st, dataset,
        priors, phystd, l2std, autodiff, physdt, extraparams,
        init_params::AbstractVector, physloglikelihood1)
        new{typeof(chain), Nothing, typeof(init_params), typeof(priors)}(dim, prob, chain,
            nothing,
            dataset, priors,
            phystd, l2std, autodiff,
            physdt, extraparams, init_params,
            physloglikelihood1)
    end
    function PDELogTargetDensity(dim, prob, chain::Lux.AbstractExplicitLayer, st, dataset,
        priors, phystd, l2std, autodiff, physdt, extraparams,
        init_params::NamedTuple, physloglikelihood1)
        new{typeof(chain), typeof(st), typeof(init_params), typeof(priors)}(dim, prob,
            chain, st,
            dataset, priors,
            phystd, l2std, autodiff,
            physdt, extraparams,
            init_params, physloglikelihood1)
    end
end

# x-mu)^2
function LogDensityProblems.logdensity(Tar::PDELogTargetDensity, θ)
    return Tar.pde_losses(θ) + Tar.bc_losses(θ) + priorweights(Tar, θ) + L2LossData(Tar, θ)
end

LogDensityProblems.dimension(Tar::PDELogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::PDELogTargetDensity)
    LogDensityProblems.LogDensityOrder{1}()
end

function generate_Tar(chain::Lux.AbstractExplicitLayer, init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    return init_params, chain, st
end

function generate_Tar(chain::Lux.AbstractExplicitLayer, init_params::Nothing)
    θ, st = Lux.setup(Random.default_rng(), chain)
    return θ, chain, st
end

function generate_Tar(chain::Flux.Chain, init_params)
    θ, re = Flux.destructure(chain)
    return init_params, re, nothing
end

function generate_Tar(chain::Flux.Chain, init_params::Nothing)
    θ, re = Flux.destructure(chain)
    # find_good_stepsize,phasepoint takes only float64
    θ = collect(Float64, θ)
    return θ, re, nothing
end

# L2 losses loglikelihood(needed mainly for ODE parameter estimation)
function L2LossData(Tar::PDELogTargetDensity, θ)
    # matrix(each row corresponds to vector u's rows)
    if isempty(Tar.dataset[end])
        return 0
    else
        nn = Tar.phi(Tar.dataset[end], θ[1:(length(θ) - Tar.extraparams)])

        L2logprob = 0
        for i in 1:length(Tar.prob.u0)
            # for u[i] ith vector must be added to dataset,nn[1,:] is the dx in lotka_volterra
            L2logprob += logpdf(MvNormal(nn[i, :], Tar.l2std[i]), Tar.dataset[i])
        end
        return L2logprob
    end
end

# priors for NN parameters + ODE constants
function priorweights(Tar::PDELogTargetDensity, θ)
    allparams = Tar.priors
    # Vector of ode parameters priors
    invpriors = allparams[2:end]

    # nn weights
    nnwparams = allparams[1]

    if Tar.extraparams > 0
        invlogpdf = sum(logpdf(invpriors[length(θ) - i + 1], θ[i])
                        for i in (length(θ) - Tar.extraparams + 1):length(θ); init = 0.0)

        return (invlogpdf
                +
                logpdf(nnwparams, θ[1:(length(θ) - Tar.extraparams)]))
    else
        return logpdf(nnwparams, θ)
    end
end

function integratorchoice(Integrator, initial_ϵ; jitter_rate = 3.0,
    tempering_rate = 3.0)
    if Integrator == JitteredLeapfrog
        Integrator(initial_ϵ, jitter_rate)
    elseif Integrator == TemperedLeapfrog
        Integrator(initial_ϵ, tempering_rate)
    else
        Integrator(initial_ϵ)
    end
end

function proposalchoice(Sampler, Integrator; n_steps = 50,
    trajectory_length = 30.0)
    if Sampler == StaticTrajectory
        Sampler(Integrator, n_steps)
    elseif Sampler == AdvancedHMC.HMCDA
        Sampler(Integrator, trajectory_length)
    else
        Sampler(Integrator)
    end
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
# lotka specific kwargs here

# NO params yet for PDE
function ahmc_bayesian_pinn_pde(pde_system, discretization;
    dataset = [[]],
    init_params = nothing, nchains = 1,
    draw_samples = 1000, l2std = [0.05],
    phystd = [0.05], priorsNNw = (0.0, 2.0),
    param = [],
    autodiff = false, physdt = 1 / 20.0f0,
    Proposal = StaticTrajectory,
    Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
    Integrator = Leapfrog,
    Metric = DiagEuclideanMetric)
    pinnrep = symbolic_discretize(pde_system, discretization)

    # for loglikelihood
    pde_loss_functions = pinnrep.loss_functions.pde_loss_functions
    bc_loss_functions = pinnrep.loss_function.bc_loss_functions
    # NN solutions for loglikelihood
    phi = pinnrep.phi
    # For sampling
    chain = discretization.chain
    discretization.additional_loss = L2LossData

    if chain isa Lux.AbstractExplicitLayer || chain isa Flux.Chain
        # Flux-vector, Lux-Named Tuple
        initial_nnθ, recon, st = generate_Tar(chain, init_params)
    else
        error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
    end

    if nchains > Threads.nthreads()
        throw(error("number of chains is greater than available threads"))
    elseif nchains < 1
        throw(error("number of chains must be greater than 1"))
    end

    if chain isa Lux.AbstractExplicitLayer
        # Lux chain(using component array later as vector_to_parameter need namedtuple,AHMC uses Float64)
        initial_θ = collect(Float64, vcat(ComponentArrays.ComponentArray(initial_nnθ)))
    else
        initial_θ = initial_nnθ
    end

    # adding ode parameter estimation
    nparameters = length(initial_θ)
    ninv = length(param)
    priors = [MvNormal(priorsNNw[1] * ones(nparameters), priorsNNw[2] * ones(nparameters))]

    # append Ode params to all paramvector
    if ninv > 0
        # shift ode params(initialise ode params by prior means)
        initial_θ = vcat(initial_θ, [Distributions.params(param[i])[1] for i in 1:ninv])
        priors = vcat(priors, param)
        nparameters += ninv
    end

    t0 = prob.tspan[1]
    # dimensions would be total no of params,initial_nnθ for Lux namedTuples
    ℓπ = PDELogTargetDensity(nparameters, prob, recon, st, dataset, priors,
        phystd, l2std, autodiff, ninv, pde_loss_functions, bc_loss_functions, phi)

    # Define Hamiltonian system
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
            initial_θ = vcat(randn(nparameters - ninv),
                initial_θ[(nparameters - ninv + 1):end])
            initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
            integrator = integratorchoice(Integrator, initial_ϵ)
            proposal = proposalchoice(Proposal, integrator)
            adaptor = Adaptor(MassMatrixAdaptor(metric),
                StepSizeAdaptor(targetacceptancerate, integrator))

            samples, stats = sample(hamiltonian, proposal, initial_θ, draw_samples, adaptor;
                progress = true, verbose = false)
            samplesc[i] = samples
            statsc[i] = stats

            mcmc_chain = Chains(hcat(samples...)')
            chains[i] = mcmc_chain
        end

        return chains, samplesc, statsc
    else
        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = integratorchoice(Integrator, initial_ϵ)
        proposal = proposalchoice(Proposal, integrator)
        adaptor = Adaptor(MassMatrixAdaptor(metric),
            StepSizeAdaptor(targetacceptancerate, integrator))

        samples, stats = sample(hamiltonian, proposal, initial_θ, draw_samples, adaptor;
            progress = true)
        # return a chain(basic chain),samples and stats
        matrix_samples = hcat(samples...)
        mcmc_chain = MCMCChains.Chains(matrix_samples')
        return mcmc_chain, samples, stats
    end
end
# try both param estim + forward solving first