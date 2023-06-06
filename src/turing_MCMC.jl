using Turing

function pinn(prob::DiffEqBase.DEProblem, parameters, dataset, recon)
    chain = recon(parameters)

    # output PINN predictions
    sol
end

myloglikelihood(x, μ) = loglikelihood(MvNormal(μ, 1), x)

# priors: pdf for W,b + pdf for ODE params
function bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain, priors, dataset, ts;
                           sampling_strategy = NUTS(0.65), num_samples = 1000,
                           syms = [Turing.@varname(theta[i]) for i in 1:length(priors)])
    param_initial, recon = Flux.destructure(chain)
    nparameters = length(param_initial)

    alpha = 0.09
    sig = sqrt(1.0 / alpha)
    @model function bayes_pinn(dataset)
        theta = Vector{T}{undef, length(priors)}
        for i in eachindex(priors)
            theta[i] ~ NamedDist(priors[i], sym[i])
        end
        nnparameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))

        preds = pinn(prob, parameters, dataset, recon)
        for i in eachindex(ts)
            datapoints ~ MvNormal(pred)
            Turing.@addlogprob! physloglikelihood(pred, μ)
        end
    end

    model = bayes_pinn(dataset)
    ch = sample(model, sampling_strategy, num_samples)
    return ch
end

function pendulum(du, u, p, t)
    ω, L = p
    x, y = u
    du[1] = y
    du[2] = -ω * y - (9.8 / L) * sin(x)
end

u0 = [1.0, 0.1]
tspan = (0.0, 10.0)
prob1 = ODEProblem(pendulum, u0, tspan, [1.0, 2.5])
