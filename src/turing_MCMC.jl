using Turing, Distributions
struct odeByNN{C, T, U}
    chain::C
    u0::U
    t0::T

    function odeByNN(re::Optimisers.Restructure, t, u0)
        new{typeof(re), typeof(t), typeof(u0)}(re, t, u0)
    end
end

function generate_phi(chain::Flux.Chain, t, u0, init_params::Nothing)
    θ, re = Flux.destructure(chain)
    odeByNN(re, t, u0), θ
end

# nn OUTPUT AT t
function (f::odeByNN{C, T, U})(t::Number,
                               θ) where {C <: Optimisers.Restructure, T, U}
    f.u0 + (t - f.t0) * first(f.chain(θ)(adapt(parameterless_type(θ), [t])))
end

function (f::odeByNN{C, T, U})(t::AbstractVector,
                               θ) where {C <: Optimisers.Restructure, T, U}
    f.u0 .+ (t .- f.t0) .* f.chain(θ)(adapt(parameterless_type(θ), t'))
end

# ODE DU/DX
function NNodederi(phi::odeByNN, t::Number, θ, autodiff::Bool)
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, θ), t)
    else
        (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, θ)) / sqrt(eps(typeof(t)))
    end
end

function NNodederi(phi::odeByNN, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, θ), t)
    else
        (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
    end
end

function physloglikelihood(chain::Any, prob::DiffEqBase.DEProblem,
                           t::AbstractVector; var = 0.5)
    u0 = prob.u0
    t0 = t[1]
    p = prob.p
    f = prob.f
    # let this be(will fix)
    autodiff = false

    phi, initparams = generate_phi(chain, t0, u0, nothing)

    μ = vec([f(phi(t[i], initparams), p, u0) for i in eachindex(t)])
    physsol = vec([NNodederi(phi, t[i], initparams, autodiff) for i in eachindex(t)])
    # print(typeof(μ))
    # print(typeof(physsol))
    # To reduce heap allocations but some erros came up
    # μ = similar(t)
    # physsol = similar(t)
    # μ = f(phi(t, initparams), p, u0)
    # physsol = NNodederi(phi, t, initparams, autodiff)
    # for i in eachindex(t)
    #     μ[i] = f(phi(t[i], initparams), p, u0)
    #     physsol[i] = NNodederi(phi, t[i], initparams, autodiff)
    # end
    return sum(abs2, (μ .- physsol) ./ (-2 * (var^2)))
    # return loglikelihood(MvNormal(physsol - μ,
    #   Diagonal(var .* ones(Float64, length(μ)))))
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
function bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain, dataset;
                           sampling_strategy = Turing.NUTS(0.65), num_samples = 1000)
    param_initial, recon = Flux.destructure(chain)
    nparameters = length(param_initial)

    if isinplace(prob)
        throw(error("The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    alpha = 0.09
    sig = sqrt(1.0 / alpha)
    # physloglikelihood(chain, prob, dataset[2], var = sig)
    DynamicPPL.@model function bayes_pinn(dataset)
        # parameter estimation?

        # prior for NN parameters(not included bias yet?) - P(Θ)
        nnparameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))
        nn = recon(nnparameters)
        preds = nn(dataset[2]')

        # # likelihood for NN pred vs Equation satif - P(phys | Θ)
        if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
            Turing.@addlogprob! physloglikelihood(nn, prob, dataset[2], var = sig)
        end

        # # likelihood for dataset vs NN pred  - P( X̄ | Θ)
        dataset[1] ~ MvNormal(vec(preds), sig .* ones(length(dataset[2])))
    end

    model = bayes_pinn(dataset)
    ch = sample(model, sampling_strategy, num_samples)
    return ch
end

# ----------need speed up
# the phase point struct
# create chain from samples,stats
# create custom distri?
# using chain with updated parameters in physloglikelihood and L2LossData

# ----------more code and compatibility
# allow options for prior,likelihood distributions
# add support for Lux chains
