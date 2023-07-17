using AdvancedHMC, ForwardDiff, LogDensityProblems, LinearAlgebra

mutable struct LogTargetDensity{C, S}
    dim::Int
    prob::DiffEqBase.DEProblem
    chain::C
    st::S
    dataset::Vector{Vector{Float64}}
    priors::Vector{Tuple{Float64, Float64}}
    phystd::Vector{Float64}
    l2std::Vector{Float64}
    autodiff::Bool
    physdt::Float64
    extraparams::Int

    function LogTargetDensity(dim, prob, chain::Optimisers.Restructure, st, dataset,
                              priors, phystd, l2std, autodiff, physdt, extraparams)
        new{typeof(chain), Nothing}(dim, prob, chain, nothing,
                                    dataset, priors,
                                    phystd, l2std, autodiff,
                                    physdt, extraparams)
    end
    function LogTargetDensity(dim, prob, chain::Lux.AbstractExplicitLayer, st, dataset,
                              priors, phystd, l2std, autodiff, physdt, extraparams)
        new{typeof(chain), typeof(st)}(dim, prob, re, st,
                                       dataset, priors,
                                       phystd, l2std, autodiff,
                                       physdt, extraparams)
    end
end

function LogDensityProblems.logdensity(Tar::LogTargetDensity, θ)
    return physloglikelihood(Tar, θ) + L2LossData(Tar, θ) + priorweights(Tar, θ)
end

LogDensityProblems.dimension(Tar::LogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::LogTargetDensity)
    LogDensityProblems.LogDensityOrder{1}()
end

function generate_Tar(chain::Lux.AbstractExplicitLayer, init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    return ComponentArrays.ComponentArray(init_params), chain, st
end

function generate_Tar(chain::Lux.AbstractExplicitLayer, init_params::Nothing)
    θ, st = Lux.setup(Random.default_rng(), chain)
    return ComponentArrays.ComponentArray(θ), chain, st
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

# nn OUTPUT AT t
function (f::LogTargetDensity{C, S})(t::AbstractVector,
                                     θ) where {C <: Optimisers.Restructure, S}
    f.prob.u0 .+ (t' .- f.prob.tspan[1]) .* f.chain(θ)(adapt(parameterless_type(θ), t'))
end

function (f::LogTargetDensity{C, S})(t::AbstractVector,
                                     θ) where {C <: Lux.AbstractExplicitLayer, S}
    # Batch via data as row vectors
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t'), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.prob.u0 .+ (t' .- f.prob.tspan[1]) .* y
end

# add similar for Lux chain
function (f::LogTargetDensity{C, S})(t::Number,
                                     θ) where {C <: Optimisers.Restructure, S}
    #  must handle paired odes hence u0 broadcasted
    f.prob.u0 .+ (t - f.prob.tspan[1]) * f.chain(θ)(adapt(parameterless_type(θ), [t]))
end

function (f::LogTargetDensity{C, S})(t::Number,
                                     θ) where {C <: Lux.AbstractExplicitLayer, S}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), [t]), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.prob.u0 .+ (t .- f.prob.tspan[1]) .* y
end

# ODE DU/DX
function NNodederi(phi::LogTargetDensity, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        # returns matrix [derivative returns vector(vector)]
        hcat(ForwardDiff.derivative.(ti -> phi(ti, θ), t)...)
    else
        (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
    end
end

# physloglike over problem timespan
function physloglikelihood(Tar::LogTargetDensity, θ)
    p = Tar.prob.p
    f = Tar.prob.f

    allparams = Tar.priors
    invparams = allparams[2:length(allparams)]
    meaninv = [invparam[1] for invparam in invparams]

    autodiff = Tar.autodiff
    dt = Tar.physdt
    t = collect(Float64, Tar.prob.tspan[1]:dt:Tar.prob.tspan[2])

    # # compare derivatives(matrix)
    out = Tar(t, θ[1:(length(θ) - Tar.extraparams)])

    # # this is a vector{vector{dx,dy}}(handle case single u(float passed))
    if length(out[:, 1]) == 1
        # shifted by prior mean
        ode_params = exp.(θ[((length(θ) - Tar.extraparams) + 1):length(θ)] + log.(meaninv))
        physsol = [f(out[:, i][1],
                     ode_params,
                     t[i])
                   for i in 1:length(out[1, :])]
    else
        # shifted by prior mean
        ode_params = exp.(θ[((length(θ) - Tar.extraparams) + 1):length(θ)] + log.(meaninv))
        physsol = [f(out[:, i],
                     ode_params,
                     t[i])
                   for i in 1:length(out[1, :])]
    end
    physsol = hcat(physsol...)

    # # convert to matrix as nnsol
    nnsol = NNodederi(Tar, t, θ[1:(length(θ) - Tar.extraparams)], autodiff)

    physlogprob = 0
    n = length(out[1, :])
    for i in 1:length(Tar.prob.u0)
        # can add phystd[i] for u[i]
        physlogprob += logpdf(MvNormal(nnsol[i, :], Tar.phystd[i]), physsol[i, :])
    end
    return physlogprob
end

# Standard L2 losses training dataset
function L2LossData(Tar::LogTargetDensity, θ)
    # matrix(each row corresponds to vector u's rows)
    nn = Tar(Tar.dataset[length(Tar.dataset)], θ[1:(length(θ) - Tar.extraparams)])

    L2logprob = 0
    n = length(nn[1, :])
    for i in 1:length(Tar.prob.u0)
        # can add l2std[i] for u[i]
        # for u[i] ith vector must be added to dataset,nn[1,:] is the dx in lotka_volterra
        L2logprob += logpdf(MvNormal(nn[i, :], Tar.l2std[i]), Tar.dataset[i])
    end
    return L2logprob
end

# priors for NN parameters + ODE constants
function priorweights(Tar::LogTargetDensity, θ)
    allparams = Tar.priors
    # ode parameters
    invparams = allparams[2:length(allparams)]
    stdinv = [invparam[2] for invparam in invparams]
    # nn weights
    nnwparams = allparams[1]
    stdw = nnwparams[2]
    prisw = nnwparams[1] .* ones(length(θ) - Tar.extraparams)

    if Tar.extraparams > 0
        return (logpdf(MvNormal(zeros(Tar.extraparams), stdinv),
                       θ[((length(θ) - Tar.extraparams) + 1):length(θ)])
                +
                logpdf(MvNormal(prisw, stdw), θ[1:(length(θ) - Tar.extraparams)]))
    else
        return logpdf(MvNormal(prisw, stdw), θ)
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
                        trajectory_length = 30)
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
function ahmc_bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain::Flux.Chain,
                                dataset::Vector{Vector{Float64}};
                                init_params = nothing, nchains = 1,
                                draw_samples = 1000, l2std = [0.05],
                                phystd = [0.05], priorsNNw = (0.0, 2.0),
                                param = [],
                                autodiff = false, physdt = 1 / 20.0f0,
                                Proposal = StaticTrajectory,
                                Adaptor = StanHMCAdaptor, targetacceptancerate = 0.75,
                                Integrator = JitteredLeapfrog,
                                Metric = DiagEuclideanMetric)

    # NN parameter prior mean and variance(PriorsNN must be a tuple)
    if isinplace(prob)
        throw(error("The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    if chain isa Lux.AbstractExplicitLayer || chain isa Flux.Chain
        initial_θ, recon, st = generate_Tar(chain, init_params)
    else
        error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
    end

    if nchains > Threads.nthreads()
        throw(error("number of chains is greater than available threads"))
    end

    # adding ode parameter estimation
    n = length(param)
    priors = [priorsNNw]

    # [i[1] for i in param]
    if length(param) > 0
        append!(initial_θ, zeros(length(param)))
        append!(priors, param)
    end
    nparameters = length(initial_θ)

    # Testing for Lux chains
    ℓπ = LogTargetDensity(nparameters, prob, recon, st, dataset, priors,
                          phystd, l2std, autodiff, physdt, n)

    #  [add f(t,θ) for t being a number]
    t0 = prob.tspan[1]
    try
        ℓπ(t0, initial_θ[1:(nparameters - n)])
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end

    # Define Hamiltonian system
    metric = Metric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)

    integrator = integratorchoice(Integrator, initial_ϵ)

    proposal = proposalchoice(Proposal, integrator)

    adaptor = Adaptor(MassMatrixAdaptor(metric),
                      StepSizeAdaptor(targetacceptancerate, integrator))

    # parallel sampling option
    if nchains != 1
        # Cache to store the chains
        chains = Vector{Any}(undef, nchains)
        statsc = Vector{Any}(undef, nchains)
        samplesc = Vector{Any}(undef, nchains)

        Threads.@threads for i in 1:nchains
            samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor;
                                    progress = true, verbose = false)
            samplesc[i] = samples
            statsc[i] = stats

            mcmc_chain = Chains(hcat(samples...)')
            chains[i] = mcmc_chain
        end

        return chains, samplesc, statsc
    else
        samples, stats = sample(hamiltonian, proposal, initial_θ, draw_samples, adaptor;
                                progress = true)
        # return a chain(basic chain),samples and stats
        matrix_samples = hcat(samples...)
        mcmc_chain = Chains(matrix_samples')
        return mcmc_chain, samples, stats
    end
end

# test for lux chins
#check if prameters estimation works(no)
# fix predictions for odes depending upon 1,p in f(u,p,t)
# lotka volterra parameters estimate
# lotka volterra learn curve beyond l2 losses

# non vectorise call functions(noticed sampling time increase)
# function NNodederi(phi::odeByNN, t::Number, θ, autodiff::Bool)
#     if autodiff
#         ForwardDiff.jacobian(t -> phi(t, θ), t)
#     else
#         (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, θ)) / sqrt(eps(typeof(t)))
#     end
# end
