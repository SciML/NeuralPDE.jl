using Test, MCMCChains
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, OptimizationOptimisers, AdvancedHMC, Lux
using Statistics, Random, Functors, ComponentArrays
using NeuralPDE, MonteCarloMeasurements

Random.seed!(110)

using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Distributions, Random

function lotka_volterra(u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    dx = (α - β * y) * x # prey
    dy = (δ * x - γ) * y # predator

    return [dx, dy]
end

# initial-value problem.
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 4.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Solve using OrdinaryDiffEq.jl solver
dt = 0.2
solution = solve(prob, Tsit5(); saveat = dt)

times = solution.t
u = hcat(solution.u...)
x = u[1, :] + (u[1, :]) .* (0.3 .* randn(length(u[1, :])))
y = u[2, :] + (u[2, :]) .* (0.3 .* randn(length(u[2, :])))
dataset = [x, y, times]

plot(times, x, label = "noisy x")
plot!(times, y, label = "noisy y")
plot!(solution, labels = ["x" "y"])

chain = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh),
    Lux.Dense(6, 2))

alg1 = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [
        Normal(1, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(0, 2)], progress = true)

alg2 = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [
        Normal(1, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(0, 2)], estim_collocate = true, progress = true)

@time sol_pestim1 = solve(prob, alg1; saveat = dt)
@time sol_pestim2 = solve(prob, alg2; saveat = dt)
plot(times, sol_pestim1.ensemblesol[1], label = "estimated x1")
plot!(times, sol_pestim2.ensemblesol[1], label = "estimated x2")
plot!(times, sol_pestim1.ensemblesol[2], label = "estimated y1")
plot!(times, sol_pestim2.ensemblesol[2], label = "estimated y2")

# comparing it with the original solution
plot!(solution, labels = ["true x" "true y"])

@show sol_pestim1.estimated_de_params
@show sol_pestim2.estimated_de_params

function fitz(u, p, t)
    v, w = u[1], u[2]
    a, b, τinv, l = p[1], p[2], p[3], p[4]

    dv = v - 0.33 * v^3 - w + l
    dw = τinv * (v + a - b * w)

    return [dv, dw]
end

prob_ode_fitzhughnagumo = ODEProblem(
    fitz, [1.0, 1.0], (0.0, 10.0), [0.7, 0.8, 1 / 12.5, 0.5])
dt = 0.5
sol = solve(prob_ode_fitzhughnagumo, Tsit5(), saveat = dt)

sig = 0.20
data = Array(sol)
dataset = [data[1, :] .+ (sig .* rand(length(sol.t))),
    data[2, :] .+ (sig .* rand(length(sol.t))), sol.t]
priors = [Normal(0.5, 1.0), Normal(0.5, 1.0), Normal(0.0, 0.5), Normal(0.5, 1.0)]

plot(sol.t, dataset[1], label = "noisy x")
plot!(sol.t, dataset[2], label = "noisy y")
plot!(sol, labels = ["x" "y"])

chain = Lux.Chain(Lux.Dense(1, 10, tanh), Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 2))

Adaptorkwargs = (Adaptor = AdvancedHMC.StanHMCAdaptor,
    Metric = AdvancedHMC.DiagEuclideanMetric, targetacceptancerate = 0.8)
alg1 = BNNODE(chain;
dataset = dataset,
draw_samples = 1000,
l2std = [0.1, 0.1],
phystd = [0.1, 0.1],
priorsNNw = (0.01, 3.0),
Adaptorkwargs = Adaptorkwargs,
param = priors, progress = true)

alg2 = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.01, 3.0),
    Adaptorkwargs = Adaptorkwargs,
    param = priors, estim_collocate = true, progress = true)

@time sol_pestim3 = solve(prob_ode_fitzhughnagumo, alg1; saveat = dt)
@time sol_pestim4 = solve(prob_ode_fitzhughnagumo, alg2; saveat = dt)
plot!(sol.t, sol_pestim3.ensemblesol[1], label = "estimated x1")
plot!(sol.t, sol_pestim4.ensemblesol[1], label = "estimated x2")
plot!(sol.t, sol_pestim3.ensemblesol[2], label = "estimated y1")
plot!(sol.t, sol_pestim4.ensemblesol[2], label = "estimated y2")

@show sol_pestim3.estimated_de_params
@show sol_pestim4.estimated_de_params
