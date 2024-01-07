using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays

Random.seed!(100)

# Cos(pit) periodic curve (Parameter Estimation)
println("Example 1, 2d Periodic System")
@parameters t, p
@variables u(..)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(p * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1)) |> Flux.f64
init1, re1 = Flux.destructure(chainf)
chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs,
    bcs,
    domains,
    [t],
    [u(t)],
    [p],
    defaults = Dict([p => 4.0]))

analytic_sol_func1(u0, t) = u0 + sin(2 * π * t) / (2 * π)
timepoints = collect(0.0:(1 / 100.0):2.0)
u = [analytic_sol_func1(0.0, timepoint) for timepoint in timepoints]
u = u .+ (u .* 0.2) .* randn(size(u))
dataset = [hcat(u, timepoints)]

# plot(dataset[1][:, 2], dataset[1][:, 1])
# plot!(timepoints, u)

# checking all training strategies
discretization = NeuralPDE.BayesianPINN([chainl],
    StochasticTraining(200),
    param_estim = true, dataset = [dataset, nothing])

ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainl],
    QuasiRandomTraining(200),
    param_estim = true, dataset = [dataset, nothing])

ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainl],
    QuadratureTraining(), param_estim = true, dataset = [dataset, nothing])

ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainl],
    GridTraining([0.02]),
    param_estim = true, dataset = [dataset, nothing])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainf],
    GridTraining([0.01]), param_estim = true, dataset = [dataset, nothing])

sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.03],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])

@test u_predict≈u_real atol=1.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3

ts = vec(sol2.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol2.ensemblesol[1])

@test u_predict≈u_real atol=0.5
@test mean(u_predict .- u_real) < 0.1
@test sol2.estimated_de_params[1]≈param atol=param * 0.3

## Example Lorenz System (Parameter Estimation)
println("Example 2, Lorenz System")
@parameters t, σ_
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (28.0 - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - 8 / 3 * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]

input_ = length(domains)
n = 7
chain = [
    Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.tanh),
        Lux.Dense(n, 1)),
    Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.tanh),
        Lux.Dense(n, 1)),
    Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.tanh),
        Lux.Dense(n, 1)),
]

#Generate Data
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5(), dt = 0.01, saveat = 0.05)
ts = sol.t
us = hcat(sol.u...)
us = us .+ ((0.05 .* randn(size(us))) .* us)
ts_ = hcat(sol(ts).t...)[1, :]
dataset = [hcat(us[i, :], ts_) for i in 1:3]

# using Plots, StatsPlots
# plot(hcat(sol.u...)[1, :], hcat(sol.u...)[2, :], hcat(sol.u...)[3, :])
# plot!(dataset[1][:, 1], dataset[2][:, 1], dataset[3][:, 1])
# plot(dataset[1][:, 2:end], dataset[1][:, 1])
# plot!(dataset[2][:, 2:end], dataset[2][:, 1])
# plot!(dataset[3][:, 2:end], dataset[3][:, 1])

discretization = NeuralPDE.BayesianPINN(chain, NeuralPDE.GridTraining([0.01]);
    param_estim = true, dataset = [dataset, nothing])

@named pde_system = PDESystem(eqs, bcs, domains,
    [t], [x(t), y(t), z(t)], [σ_], defaults = Dict([p => 1.0 for p in [σ_]]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 50,
    bcstd = [0.3, 0.3, 0.3],
    phystd = [0.1, 0.1, 0.1],
    l2std = [1, 1, 1],
    priorsNNw = (0.0, 1.0),
    saveats = [0.01],
    param = [Normal(12.0, 2)])

idealp = 10.0
p_ = sol1.estimated_de_params[1]

# plot(pmean(sol1.ensemblesol[1]), pmean(sol1.ensemblesol[2]), pmean(sol1.ensemblesol[3]))
# plot(sol1.timepoints[1]', pmean(sol1.ensemblesol[1]))
# plot!(sol1.timepoints[2]', pmean(sol1.ensemblesol[2]))
# plot!(sol1.timepoints[3]', pmean(sol1.ensemblesol[3]))

@test sum(abs, pmean(p_) - 10.00) < 0.3 * idealp[1]
# @test sum(abs, pmean(p_[2]) - (8 / 3)) < 0.3 * idealp[2]