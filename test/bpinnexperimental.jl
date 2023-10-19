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
dt = 0.01
solution = solve(prob, Tsit5(); saveat = dt)

times = solution.t
u = hcat(solution.u...)
x = u[1, :] + (u[1, :]) .* (0.05 .* randn(length(u[1, :])))
y = u[2, :] + (u[2, :]) .* (0.05 .* randn(length(u[2, :])))
dataset = [x, y, times]

plot(times, x, label = "noisy x")
plot!(times, y, label = "noisy y")
plot!(solution, labels = ["x" "y"])

chain = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh),
    Lux.Dense(6, 2))

alg = BNNODE(chain;
dataset = dataset,
draw_samples = 1000,
l2std = [0.1, 0.1],
phystd = [0.1, 0.1],
priorsNNw = (0.0, 3.0),
param = [
    Normal(1, 2),
    Normal(2, 2),
    Normal(2, 2),
    Normal(0, 2)], progress = false)

sol_pestim = solve(prob, alg; saveat = dt)
plot(times, sol_pestim.ensemblesol[1], label = "estimated x")
plot!(times, sol_pestim.ensemblesol[2], label = "estimated y")

# comparing it with the original solution
plot!(solution, labels = ["true x" "true y"])

sol_pestim.estimated_ode_params