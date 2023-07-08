using AdvancedHMC, ForwardDiff, LogDensityProblems
using LinearAlgebra

mutable struct LogTargetDensity{C, S}
    dim::Int
    prob::DiffEqBase.DEProblem
    chain::C
    st::S
    dataset::Tuple{AbstractVector, AbstractVector}
    priorsNN::Tuple{Float64, Float64}
    phystd::Float64
    l2std::Float64
    autodiff::Bool
    physdt::Float64

    function LogTargetDensity(dim, prob, chain::Optimisers.Restructure, st, dataset,
        priorsNN, phystd, l2std, autodiff, physdt)
        new{typeof(chain), Nothing}(dim, prob, chain, nothing,
            dataset, priorsNN,
            phystd, l2std, autodiff,
            physdt)
    end
    function LogTargetDensity(dim, prob, chain::Lux.AbstractExplicitLayer, st, dataset,
        priorsNN, phystd, l2std, autodiff, physdt)
        new{typeof(chain), typeof(st)}(dim, prob, re, st,
            dataset, priorsNN,
            phystd, l2std, autodiff,
            physdt)
    end
end

function LogDensityProblems.logdensity(Tar::LogTargetDensity, θ)
    return physloglikelihood(Tar, θ) + L2LossData(Tar, θ) + priorweights(Tar, θ)
end

LogDensityProblems.dimension(Tar::LogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::LogTargetDensity)
    LogDensityProblems.LogDensityOrder{0}()
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
    # if init_params==nothing
    #     θ = collect(Float64, θ)
    #     return θ, re, nothing
    # else 
    #     return init_params, re, nothing
    # end
    return init_params, re, nothing
end
function generate_Tar(chain::Flux.Chain, init_params::Nothing)
    θ, re = Flux.destructure(chain)
    # find_good_stepsize takes only float64?
    θ = collect(Float64, θ)
    return θ, re, nothing
end

# nn OUTPUT AT t
function (f::LogTargetDensity{C, S})(t::AbstractVector,
    θ) where {C <: Optimisers.Restructure, S}
    f.prob.u0 .+ (t .- f.prob.tspan[1]) .* vec(f.chain(θ)(t'))
end

function (f::LogTargetDensity{C, S})(t::AbstractVector,
    θ) where {C <: Lux.AbstractExplicitLayer, S}
    # Batch via data as row vectors
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t'), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.u0 .+ (t' .- f.t0) .* y
end

# ODE DU/DX
function NNodederi(phi::LogTargetDensity, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, θ), t)
    else
        (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
    end
end

function physloglikelihood(Tar::LogTargetDensity, θ)
    p = Tar.prob.p
    f = Tar.prob.f
    var = Tar.phystd^2
    autodiff = Tar.autodiff
    dt = Tar.physdt
    t = collect(Float64, Tar.prob.tspan[1]:dt:Tar.prob.tspan[2])

    # compare derivatives
    out = Tar(t, θ)
    physsol = [f(out[i], p, t[i]) for i in eachindex(out)]
    nnsol = NNodederi(Tar, t, θ, autodiff)

    # distribution's mean is forwarddiff diag(jacobian)
    if autodiff
        nnsol = diag(nnsol)
    end

    n = length(nnsol)

    return logpdf(MvNormal(nnsol, Diagonal(var .* ones(n))), physsol)
end

# standard MvNormal Dist Assume
function L2LossData(Tar::LogTargetDensity, θ)
    nn = Tar(Tar.dataset[2], θ)
    n = length(nn)
    var = Tar.l2std^2
    return logpdf(MvNormal(nn, Diagonal(var .* ones(n))), Tar.dataset[1])
end

function priorweights(Tar::LogTargetDensity, θ)
    params = Tar.priorsNN
    return logpdf(MvNormal(θ, Diagonal(params[2]^2 .* ones(length(θ)))),
        params[1] * ones(length(θ)))
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
function ahmc_bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain::Flux.Chain,
    dataset::Tuple{AbstractVector, AbstractVector};
    init_params = nothing,
    draw_samples = 1000, l2std = 0.08,
    phystd = 0.08, priorsNN = (0, 2), autodiff = false,
    physdt = 1 / 20.0f0,
    Proposal = AdvancedHMC.NUTS{MultinomialTS,
        GeneralisedNoUTurn},
    Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
    Integrator = Leapfrog, Metric = DiagEuclideanMetric)
    # NN parameter prior mean and variance(PriorsNN must be a tuple)
    if isinplace(prob)
        throw(error("The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    if chain isa Lux.AbstractExplicitLayer || chain isa Flux.Chain
        initial_θ, recon, st = generate_Tar(chain, init_params)
    else
        error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
    end

    # adding ode parameter estimation?
    nparameters = length(initial_θ)
    ℓπ = LogTargetDensity(nparameters, prob, recon, st, dataset, priorsNN,
        phystd, l2std, autodiff, physdt)

    #  [add f(t,θ) for t being a number]
    # try
    #     ℓπ(t0, initial_θ)
    # catch err
    #     if isa(err, DimensionMismatch)
    #         throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
    #     else
    #         throw(err)
    #     end
    # end

    n_samples = draw_samples
    metric = Metric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)

    # choices for integrators?
    # [define n for JL and a for TL]
    # if Integrator == JitteredLeapfrog(n)
    #     integrator = JitteredLeapfrog(initial_ϵ, n)
    # elseif Integrator == TemperedLeapfrog(a)
    #     integrator == TemperedLeapfrog(initial_ϵ, a)
    # else
    #     integrator = Leapfrog(initial_ϵ)
    # end

    integrator = Leapfrog(initial_ϵ)
    proposal = Proposal(integrator)
    adaptor = Adaptor(MassMatrixAdaptor(metric),
        StepSizeAdaptor(targetacceptancerate, integrator))

    samples, stats = sample(hamiltonian, proposal, initial_θ,
        n_samples, adaptor;
        progress = true)

    # return a chain(basic chain),samples and stats
    matrix_samples = hcat(samples...)
    mcmc_chain = Chains(matrix_samples')
    return mcmc_chain, samples, stats
end

# non vectorised functions(i noticed sampling time increase)
# function NNodederi(phi::odeByNN, t::Number, θ, autodiff::Bool)
#     if autodiff
#         ForwardDiff.jacobian(t -> phi(t, θ), t)
#     else
#         (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, θ)) / sqrt(eps(typeof(t)))
#     end
# end

# function (f::odeByNN{C, T, U})(t::Number,
#                                θ) where {C <: Optimisers.Restructure, T, U}
#     f.u0 + (t - f.t0) * first(f.chain(θ)(adapt(parameterless_type(θ), [t])))
# end
