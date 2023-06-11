using Turing

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
function (f::odeByNN{C, T, U})(t::AbstractVector,
                               θ) where {C <: Optimisers.Restructure, T, U}
    f.u0 .+ (t - f.t0) * f.chain(θ)(adapt(parameterless_type(θ), t'))
end

# ODE DU/DX
function NNodederi(phi::odeByNN, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, θ), t)
    else
        (phi(t + sqrt(eps(typeof(t)))) - phi(t, θ)) / sqrt(eps(typeof(t)))
    end
end

function physloglikelihood(prob::DiffEqBase.DEProblem, pred, dataset; var = 0.5)
    u0 = prob.u0
    tspan = prob.tspan
    t0 = tspan[1]
    p = prob.p
    f = prob.f

    autodiff = false
    phi, initparams = generate_phi(chain, t0, u0, nothing)
    μ = [f(phi(t, θ), p, u0) for t in tspan]
    physsol = NNodederi(phi, t, θ, autodiff)
    loglikelihood(MvNormal(μ, var), physsol)
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
function bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain, dataset;
                           sampling_strategy = NUTS(0.65), num_samples = 1000)
    param_initial, recon = Flux.destructure(chain)
    nparameters = length(param_initial)

    alpha = 0.09
    sig = sqrt(1.0 / alpha)
    @model function bayes_pinn(dataset)
        # parameter estimation?

        # prior for NN parameters(not included bias yet?)
        nnparameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))
        nn = recon(nnparameters)
        preds = nn(dataset[:2])

        # likelihood for dataset vs NN pred
        dataset[:1] ~ MvNormal(preds, sig .* ones(length(dataset[:2])))

        # likelihood for NN pred vs Equation satif
        if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
            Turing.@addlogprob! physloglikelihood(prob, pred, dataset, var = sig)
        end
    end

    model = bayes_pinn(dataset)
    ch = sample(model, sampling_strategy, num_samples)
    return ch
end

# Testing out Code (will put in bpinntests.jl file in tests directory)
function pendulum(du, u, p, t)
    ω, L = p
    x, y = u
    du[1] = y
    du[2] = -ω * y - (9.8 / L) * sin(x)
end

u0 = [1.0, 0.1]
tspan = (0.0, 10.0)

p = [1.0, 2.5]
prob1 = ODEProblem(pendulum, u0, tspan, p)
nnt = Chain(Dense(1, 3, tanh), Dense(3, 4, tanh), Dense(4, 1, sigmoid))
dataset = []
bayesian_pinn_ode(prob1, nnt, dataset)