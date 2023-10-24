# # Testing Code
using Test, MCMCChains
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, OptimizationOptimisers, AdvancedHMC, Lux
using Statistics, Random, Functors, ComponentArrays
using NeuralPDE, MonteCarloMeasurements

# note that current testing bounds can be easily further tightened but have been inflated for support for Julia build v1
# on latest Julia version it performs much better for below tests
Random.seed!(100)

# for sampled params->lux ComponentArray
function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return Functors.fmap(get_ps, ps)
end

## PROBLEM-1 (WITHOUT PARAMETER ESTIMATION)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)
tspan = (0.0, 2.0)
u0 = 0.0
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)
p = prob.p

# Numerical and Analytical Solutions: testing ahmc_bayesian_pinn_ode()
ta = range(tspan[1], tspan[2], length = 300)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
time = vec(collect(Float64, ta))
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# testing points for solve() call must match saveat(1/50.0) arg
ta0 = range(tspan[1], tspan[2], length = 101)
u1 = [linear_analytic(u0, nothing, ti) for ti in ta0]
x̂1 = collect(Float64, Array(u1) + 0.02 * randn(size(u1)))
time1 = vec(collect(Float64, ta0))
physsol0_1 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

chainflux = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1)) |> Flux.f64
chainlux = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
init1, re1 = destructure(chainflux)
θinit, st = Lux.setup(Random.default_rng(), chainlux)

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux,
    draw_samples = 2500)

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux,
    draw_samples = 2500)

# can change training strategies by adding this to call (Quadratuer and GridTraining show good results but stochastics sampling techniques perform bad)
# strategy = QuadratureTraining(; quadrature_alg = QuadGKJL(),
#     reltol = 1e-6,
#     abstol = 1e-3, maxiters = 1000,
#     batch = 0)

alg = NeuralPDE.BNNODE(chainflux, draw_samples = 2500)
sol1flux = solve(prob, alg)

alg = NeuralPDE.BNNODE(chainlux, draw_samples = 2500)
sol1lux = solve(prob, alg)

# testing points
t = time
# Mean of last 500 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
out = re1.(fhsamples1[(end - 500):end])
yu = collect(out[i](t') for i in eachindex(out))
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

θ = [vector_to_parameters(fhsamples1[i], θinit) for i in 2000:2500]
luxar = [chainlux(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

# --------------------- ahmc_bayesian_pinn_ode() call
@test mean(abs.(x̂ .- meanscurve1)) < 0.05
@test mean(abs.(physsol1 .- meanscurve1)) < 0.005
@test mean(abs.(x̂ .- meanscurve2)) < 0.05
@test mean(abs.(physsol1 .- meanscurve2)) < 0.005

#--------------------- solve() call 
@test mean(abs.(x̂1 .- sol1flux.ensemblesol[1])) < 0.05
@test mean(abs.(physsol0_1 .- sol1flux.ensemblesol[1])) < 0.05
@test mean(abs.(x̂1 .- sol1lux.ensemblesol[1])) < 0.05
@test mean(abs.(physsol0_1 .- sol1lux.ensemblesol[1])) < 0.05

## PROBLEM-1 (WITH PARAMETER ESTIMATION)
linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
linear = (u, p, t) -> cos(p * t)
tspan = (0.0, 2.0)
u0 = 0.0
p = 2 * pi
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, p)

# Numerical and Analytical Solutions
sol1 = solve(prob, Tsit5(); saveat = 0.01)
u = sol1.u
time = sol1.t

# BPINN AND TRAINING DATASET CREATION(dataset must be defined only inside problem timespan!)
ta = range(tspan[1], tspan[2], length = 100)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# testing points for solve call(saveat=1/50.0 ∴ at t = collect(eltype(saveat), prob.tspan[1]:saveat:prob.tspan[2] internally estimates)
ta0 = range(tspan[1], tspan[2], length = 101)
u1 = [linear_analytic(u0, p, ti) for ti in ta0]
x̂1 = collect(Float64, Array(u1) + 0.2 * randn(size(u1)))
time1 = vec(collect(Float64, ta0))
physsol1_1 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

chainflux1 = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1)) |> Flux.f64
chainlux1 = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
init1, re1 = destructure(chainflux1)
θinit, st = Lux.setup(Random.default_rng(), chainlux1)

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
    dataset = dataset,
    draw_samples = 2500,
    physdt = 1 / 50.0,
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(9,
            0.5),
    ])

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux1,
    dataset = dataset,
    draw_samples = 2500,
    physdt = 1 / 50.0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)])

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 2500, physdt = 1 / 50.0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)])

sol2flux = solve(prob, alg)

alg = NeuralPDE.BNNODE(chainlux1, dataset = dataset,
    draw_samples = 2500,
    physdt = 1 / 50.0,
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(9,
            0.5),
    ])

sol2lux = solve(prob, alg)

# testing points
t = time
# Mean of last 500 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
out = re1.([fhsamples1[i][1:22] for i in 2000:2500])
yu = collect(out[i](t') for i in eachindex(out))
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

θ = [vector_to_parameters(fhsamples2[i][1:(end - 1)], θinit) for i in 2000:2500]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

# --------------------- ahmc_bayesian_pinn_ode() call  
@test mean(abs.(physsol1 .- meanscurve1)) < 0.15
@test mean(abs.(physsol1 .- meanscurve2)) < 0.15

# ESTIMATED ODE PARAMETERS (NN1 AND NN2)
@test abs(p - mean([fhsamples2[i][23] for i in 2000:2500])) < abs(0.35 * p)
@test abs(p - mean([fhsamples1[i][23] for i in 2000:2500])) < abs(0.35 * p)

#-------------------------- solve() call  
@test mean(abs.(physsol1_1 .- sol2flux.ensemblesol[1])) < 8e-2
@test mean(abs.(physsol1_1 .- sol2lux.ensemblesol[1])) < 8e-2

# ESTIMATED ODE PARAMETERS (NN1 AND NN2)
@test abs(p - sol2flux.estimated_ode_params[1]) < abs(0.15 * p)
@test abs(p - sol2lux.estimated_ode_params[1]) < abs(0.15 * p)

## PROBLEM-2
linear = (u, p, t) -> u / p + exp(t / p) * cos(t)
tspan = (0.0, 10.0)
u0 = 0.0
p = -5.0
prob = ODEProblem(linear, u0, tspan, p)
linear_analytic = (u0, p, t) -> exp(t / p) * (u0 + sin(t))

# SOLUTION AND CREATE DATASET
sol = solve(prob, Tsit5(); saveat = 0.1)
u = sol.u
time = sol.t
x̂ = u .+ (u .* 0.2) .* randn(size(u))
dataset = [x̂, time]
t = sol.t
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]

ta0 = range(tspan[1], tspan[2], length = 501)
u1 = [linear_analytic(u0, p, ti) for ti in ta0]
time1 = vec(collect(Float64, ta0))
physsol2 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

chainflux12 = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh),
    Flux.Dense(6, 1)) |> Flux.f64
chainlux12 = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
init1, re1 = destructure(chainflux12)
θinit, st = Lux.setup(Random.default_rng(), chainlux12)

fh_mcmc_chainflux12, fhsamplesflux12, fhstatsflux12 = ahmc_bayesian_pinn_ode(prob,
    chainflux12,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [
        0.03],
    priorsNNw = (0.0,
        10.0))

fh_mcmc_chainflux22, fhsamplesflux22, fhstatsflux22 = ahmc_bayesian_pinn_ode(prob,
    chainflux12,
    dataset = dataset,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [
        0.03,
    ],
    priorsNNw = (0.0,
        10.0),
    param = [
        Normal(-7,
            4),
    ])

fh_mcmc_chainlux12, fhsampleslux12, fhstatslux12 = ahmc_bayesian_pinn_ode(prob, chainlux12,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [0.03],
    priorsNNw = (0.0,
        10.0))

fh_mcmc_chainlux22, fhsampleslux22, fhstatslux22 = ahmc_bayesian_pinn_ode(prob, chainlux12,
    dataset = dataset,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [0.03],
    priorsNNw = (0.0,
        10.0),
    param = [
        Normal(-7,
            4),
    ])

alg = NeuralPDE.BNNODE(chainflux12,
    dataset = dataset,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [
        0.03,
    ],
    priorsNNw = (0.0,
        10.0),
    param = [
        Normal(-7,
            4),
    ])

sol3flux_pestim = solve(prob, alg)

alg = NeuralPDE.BNNODE(chainlux12,
    dataset = dataset,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [0.03],
    priorsNNw = (0.0,
        10.0),
    param = [
        Normal(-7,
            4),
    ])

sol3lux_pestim = solve(prob, alg)

# testing timepoints
t = sol.t
#------------------------------ ahmc_bayesian_pinn_ode() call 
# Mean of last 500 sampled parameter's curves(flux chains)[Ensemble predictions]
out = re1.([fhsamplesflux12[i][1:61] for i in 1000:1500])
yu = [out[i](t') for i in eachindex(out)]
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1_1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

out = re1.([fhsamplesflux22[i][1:61] for i in 1000:1500])
yu = [out[i](t') for i in eachindex(out)]
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1_2 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

@test mean(abs.(sol.u .- meanscurve1_1)) < 1e-2
@test mean(abs.(physsol1 .- meanscurve1_1)) < 1e-2
@test mean(abs.(sol.u .- meanscurve1_2)) < 5e-2
@test mean(abs.(physsol1 .- meanscurve1_2)) < 5e-2

# estimated parameters(flux chain)
param1 = mean(i[62] for i in fhsamplesflux22[1000:1500])
@test abs(param1 - p) < abs(0.3 * p)

# Mean of last 500 sampled parameter's curves(lux chains)[Ensemble predictions]
θ = [vector_to_parameters(fhsampleslux12[i], θinit) for i in 1000:1500]
luxar = [chainlux12(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

θ = [vector_to_parameters(fhsampleslux22[i][1:(end - 1)], θinit) for i in 1000:1500]
luxar = [chainlux12(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs.(sol.u .- meanscurve2_1)) < 1e-1
@test mean(abs.(physsol1 .- meanscurve2_1)) < 1e-1
@test mean(abs.(sol.u .- meanscurve2_2)) < 5e-2
@test mean(abs.(physsol1 .- meanscurve2_2)) < 5e-2

# estimated parameters(lux chain)
param1 = mean(i[62] for i in fhsampleslux22[1000:1500])
@test abs(param1 - p) < abs(0.3 * p)

#-------------------------- solve() call 
# (flux chain)
@test mean(abs.(physsol2 .- sol3flux_pestim.ensemblesol[1])) < 0.15
# estimated parameters(flux chain)
param1 = sol3flux_pestim.estimated_ode_params[1]
@test abs(param1 - p) < abs(0.45 * p)

# (lux chain)
@test mean(abs.(physsol2 .- sol3lux_pestim.ensemblesol[1])) < 0.15
# estimated parameters(lux chain)
param1 = sol3lux_pestim.estimated_ode_params[1]
@test abs(param1 - p) < abs(0.45 * p)

# ______________________________________________PDE_BPINN_SOLVER_________________________________________________________________

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
using Plots, OrdinaryDiffEq, Distributions, Random
import ModelingToolkit: Interval, infimum, supremum

# @parameters t
# @variables x(..) y(..)

# α, β, γ, δ = [1.5, 1.0, 3.0, 1.0]

# Dt = Differential(t)
# eqs = [Dt(x(t)) ~ (α - β * y(t)) * x(t), Dt(y(t)) ~ (δ * x(t) - γ) * y(t)]
# bcs = [x(0) ~ 1.0, y(0) ~ 1.0]
# domains = [t ∈ Interval(0.0, 6.0)]

# chain = [Flux.Chain(Flux.Dense(1, 4, tanh), Flux.Dense(4, 4),
#         Flux.Dense(4, 1)), Flux.Chain(Flux.Dense(1, 4, tanh), Flux.Dense(4, 4),
#         Flux.Dense(4, 1))]

# chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh),
#     Flux.Dense(6, 2))

# init, re = destructure(chainf)
# init1, re1 = destructure(chain[1])
# init2, re2 = destructure(chain[2])
# chainf = re(ones(size(init)))
# chain[1] = re1(ones(size(init1)))
# chain[2] = re2(ones(size(init2)))

# discretization = NeuralPDE.PhysicsInformedNN(chain,
#     GridTraining([0.01]))
# @named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t)])

# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system, discretization;
#     draw_samples = 100,
#     bcstd = [0.1, 0.1],
#     phystd = [0.1, 0.1], priorsNNw = (0.0, 10.0), progress = true)

# # FLUX CHAIN post sampling
# tspan = (0.0, 6.0)
# t1 = collect(tspan[1]:0.01:tspan[2])
# out1 = re1.([samples[i][1:33]
#              for i in 80:100])
# out2 = re2.([samples[i][34:end]
#              for i in 80:100])

# luxar1 = collect(out1[i](t1') for i in eachindex(out1))
# luxar2 = collect(out2[i](t1') for i in eachindex(out2))
# plot(t1, luxar1[end]')
# plot!(t1, luxar2[end]')

# # LUX CHAIN post sampling
# θinit, st = Lux.setup(Random.default_rng(), chain[1])
# θinit1, st1 = Lux.setup(Random.default_rng(), chain[2])

# θ1 = [vector_to_parameters(samples[i][1:22], θinit) for i in 50:100]
# θ2 = [vector_to_parameters(samples[i][23:end], θinit1) for i in 50:100]
# tspan = (0.0, 6.0)
# t1 = collect(tspan[1]:0.01:tspan[2])
# luxar1 = [chain[1](t1', θ1[i], st)[1] for i in 1:50]
# luxar2 = [chain[2](t1', θ2[i], st1)[1] for i in 1:50]

# plot(t1, luxar1[500]')
# plot!(t1, luxar2[500]')

# # BPINN 0DE SOLVER COMPARISON CASE
# function lotka_volterra(u, p, t)
#     # Model parameters.
#     α, β, γ, δ = p
#     # Current state.
#     x, y = u

#     # Evaluate differential equations.
#     dx = (α - β * y) * x # prey
#     dy = (δ * x - γ) * y # predator

#     return [dx, dy]
# end

# # initial-value problem.
# u0 = [1.0, 1.0]
# p = [1.5, 1.0, 3.0, 1.0]
# tspan = (0.0, 6.0)
# prob = ODEProblem(lotka_volterra, u0, tspan, p)

# cospit example
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
initf, re = destructure(chainf)

chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

# non adaptive case
discretization = NeuralPDE.PhysicsInformedNN(chainf, GridTraining([0.01]))
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.01],
    phystd = [0.01],
    priorsNNw = (0.0, 10.0),
    progress = true)

discretization = NeuralPDE.PhysicsInformedNN(chainl, GridTraining([0.01]))
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.01],
    phystd = [0.01],
    priorsNNw = (0.0, 10.0),
    progress = true)

tspan = (0.0, 4.0)
t1 = collect(tspan[1]:0.01:tspan[2])

out1 = re.([samples[i] for i in 800:1000])
luxar1 = collect(out1[i](t1') for i in eachindex(out1))

transsamples = [vector_to_parameters(sample, initl) for sample in samples]
luxar2 = [chainl(t1', transsamples[i], st)[1] for i in 800:1000]

yu = [linear_analytic(0, nothing, t) for t in t1]
plot(t1, yu)
plot!(t1, luxar1[end]')
plot!(t1, luxar2[end]')
