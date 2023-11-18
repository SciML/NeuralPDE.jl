using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements

# Forward solving example
@parameters t
@variables u(..)

Dt = Differential(t)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(2 * π * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 4.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1))
init1, re1 = destructure(chainf)
chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

# non adaptive case
discretization = NeuralPDE.PhysicsInformedNN(chainf, GridTraining([0.01]))
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.02],
    phystd = [0.01],
    priorsNNw = (0.0, 1.0),
    progress = true)

discretization = NeuralPDE.PhysicsInformedNN(chainl, GridTraining([0.01]))
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.02],
    phystd = [0.01],
    priorsNNw = (0.0, 1.0),
    progress = true)

tspan = (0.0, 4.0)
t1 = collect(tspan[1]:0.01:tspan[2])

out1 = re.([samples[i] for i in 800:1000])
luxar1 = collect(out1[i](t1') for i in eachindex(out1))
fluxmean = [mean(vcat(luxar1...)[:, i]) for i in eachindex(t1)]

transsamples = [vector_to_parameters(sample, initl) for sample in samples]
luxar2 = [chainl(t1', transsamples[i], st)[1] for i in 800:1000]
luxmean = [mean(vcat(luxar2...)[:, i]) for i in eachindex(t1)]

u = [linear_analytic(0, nothing, t) for t in t1]

@test mean(abs.(u .- fluxmean)) < 5e-2
@test mean(abs.(u .- luxmean)) < 5e-2

# Parameter estimation example
@parameters t p
@variables u(..)

Dt = Differential(t)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)
eqs = Dt(u(t)) - cos(p * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 4.0)]
p = 2 * π

chainf = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 1))
init1, re1 = destructure(chainf)
chainl = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)], [p],
    defaults = Dict(p => 3))

ta = range(0.0, 4.0, length = 50)
u = [linear_analytic(0.0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.2 .* Array(u) .* randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]

discretization = NeuralPDE.PhysicsInformedNN([chainf],
    GridTraining(0.01),
    param_estim = true)

mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500, physdt = 1 / 20.0,
    bcstd = [1.0],
    phystd = [0.01], l2std = [0.01],
    param = [Normal(9, 2)],
    priorsNNw = (0.0, 10.0),
    dataset = dataset,
    progress = true)

discretization = NeuralPDE.PhysicsInformedNN([chainl],
    GridTraining(0.01),
    param_estim = true)

mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.1],
    phystd = [0.01], l2std = [0.01], param = [LogNormal(4, 2)],
    priorsNNw = (0.0, 10.0),
    dataset = dataset,
    progress = true)

tspan = (0.0, 4.0)
t1 = collect(tspan[1]:0.01:tspan[2])

out1 = re.([samples[i][1:(end - 1)] for i in 1300:1500])
luxar1 = collect(out1[i](t1') for i in eachindex(out1))
fluxmean = [mean(vcat(luxar1...)[:, i]) for i in eachindex(t1)]

transsamples = [vector_to_parameters(sample, initl) for sample[1:(end - 1)] in samples]
luxar2 = [chainl(t1', transsamples[i], st)[1] for i in 1300:1500]
luxmean = [mean(vcat(luxar2...)[:, i]) for i in eachindex(t1)]

u = [linear_analytic(0, nothing, t) for t in t1]

@test mean(abs.(u .- fluxmean)) < 5e-2
@test mean(abs.(u .- luxmean)) < 5e-2

@test mean(p .- [samples[i][end] for i in 1300:1500]) < 0.4 * p
@test mean(p .- [samples[i][end] for i in 1300:1500]) < 0.4 * p