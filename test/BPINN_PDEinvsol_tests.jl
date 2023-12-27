using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays

# Cospit example
@parameters t, p
@variables u(..)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(p * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1))
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

# non adaptive case
discretization = NeuralPDE.PhysicsInformedNN([chainl],
    GridTraining([0.02]),
    param_estim = true)

analytic_sol_func1(u0, t) = u0 + sin(2 * π * t) / (2 * π)
timepoints = collect(0.0:(1 / 100.0):2.0)
u = [analytic_sol_func(0.0, timepoint) for timepoint in timepoints]
u = u .+ (u .* 0.2) .* randn(size(u))
dataset = [hcat(u, timepoints)]

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.02], l2std = [0.005],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(4.0, 5.0)],
    dataset = dataset)

discretization = NeuralPDE.PhysicsInformedNN([chainf],
    GridTraining([0.01]),
    param_estim = true)

sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.02], l2std = [0.005],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(4.0, 5.0)],
    dataset = dataset)

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])
@test u_predict≈u_real atol=0.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3

ts = vec(sol2.timepoints[1])
u_real = [analytic_sol_func(0.0, t) for t in ts]
u_predict = pmean(sol2.ensemblesol[1])
@test u_predict≈u_real atol=0.5
@test mean(u_predict .- u_real) < 0.1
@test sol2.estimated_de_params[1]≈param atol=param * 0.3