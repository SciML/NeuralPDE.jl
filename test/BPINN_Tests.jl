# Testing Code
using DifferentialEquations, MCMCChains, ForwardDiff, Distributions
using NeuralPDE, Flux, OptimizationOptimisers, AdvancedHMC, Lux
using StatProfilerHTML, Profile, Statistics, Random, Functors, ComponentArrays
using BenchmarkTools, Test

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

# PROBLEM-1 (WITHOUT PARAMETER ESTIMATION)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)
tspan = (0.0, 2.0)
u0 = 0.0
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# Numerical and Analytical Solutions
ta = range(tspan[1], tspan[2], length = 300)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
sol1 = solve(prob, Tsit5())

# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂[1:100], time[1:100]]

# Call BPINN, create chain
chainflux = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1))
chainlux = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux,
                                                              dataset = dataset,
                                                              draw_samples = 2000)
fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux,
                                                              dataset = dataset,
                                                              draw_samples = 2000)

init1, re1 = destructure(chainflux)
θinit, st = Lux.setup(Random.default_rng(), chainlux)

# TESTING TIMEPOINTS TO PLOT ON,Actual Sols and actual data
t = time
p = prob.p
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]

# Mean of last 1000 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
out = re1.(fhsamples1[(end - 1000):end])
yu = collect(out[i](t') for i in eachindex(out))
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

θ = [vector_to_parameters(fhsamples2[i], θinit) for i in 1000:2000]
luxar = [chainlux(t', θ[i], st)[1] for i in 1:1000]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(x̂ .- meanscurve1)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve1)) < 1e-5
@test mean(abs2.(x̂ .- meanscurve2)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve2)) < 1e-5

# PROBLEM-1 (WITH PARAMETER ESTIMATION)
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

# BPINN AND TRAINING DATASET CREATION
ta = range(tspan[1], tspan[2], length = 200)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂[1:50], time[1:50]]

# comparing how diff NNs capture non-linearity
chainflux1 = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1))
chainlux1 = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                              dataset = dataset,
                                                              draw_samples = 2000,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 2)],
                                                              Metric = DiagEuclideanMetric)

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux1,
                                                              dataset = dataset,
                                                              draw_samples = 2000,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 2)],
                                                              Metric = DiagEuclideanMetric)

init1, re1 = destructure(chainflux1)
θinit, st = Lux.setup(Random.default_rng(), chainlux1)

#   PLOT testing points
t = time
p = prob.p
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]

# Mean of last 1000 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
out = re1.([fhsamples1[i][1:22] for i in 1000:2000])
yu = collect(out[i](t') for i in eachindex(out))
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

θ = [vector_to_parameters(fhsamples2[i][1:(end - 1)], θinit) for i in 1000:2000]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:1000]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(x̂ .- meanscurve1)) < 2e-2
@test mean(abs2.(physsol1 .- meanscurve1)) < 2e-2
@test mean(abs2.(x̂ .- meanscurve2)) < 3e-3
@test mean(abs2.(physsol1 .- meanscurve2)) < 2e-3

# ESTIMATED ODE PARAMETERS (NN1 AND NN2)
@test abs(p - mean([fhsamples2[i][23] for i in 1000:2000])) < 0.1 * p
@test abs(p - mean([fhsamples1[i][23] for i in 1000:2000])) < 0.2 * p

# PROBLEM-2
linear = (u, p, t) -> -u / p[1] + exp(t / p[2]) * cos(t)
tspan = (0.0, 10.0)
u0 = 0.0
p = [5.0, -5.0]
prob = ODEProblem(linear, u0, tspan, p)

# PROBLEM-2
linear = (u, p, t) -> -u / 5 + exp(-t / 5) * cos(t)
linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))

# PLOT SOLUTION AND CREATE DATASET
sol = solve(prob, Tsit5(); saveat = 0.05)
u = sol.u[1:100]
time = sol.t[1:100]

# dataset and BPINN create
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
dataset = [x̂, time]

chainflux1 = Flux.Chain(Flux.Dense(1, 5, tanh), Flux.Dense(5, 5, tanh), Flux.Dense(5, 1))
chainlux1 = Lux.Chain(Lux.Dense(1, 5, tanh), Lux.Dense(5, 5, tanh), Lux.Dense(5, 1))

fh_mcmc_chainflux1, fhsamplesflux1, fhstatsflux1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                                          dataset = dataset,
                                                                          draw_samples = 1000,
                                                                          l2std = [0.05],
                                                                          phystd = [0.05],
                                                                          priorsNNw = (0.0,
                                                                                       3.0))

fh_mcmc_chainflux2, fhsamplesflux2, fhstatsflux2 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                                          dataset = dataset,
                                                                          draw_samples = 1000,
                                                                          l2std = [0.05],
                                                                          phystd = [0.05],
                                                                          priorsNNw = (0.0,
                                                                                       3.0),
                                                                          param = [
                                                                              Normal(6.5,
                                                                                     2),
                                                                              Normal(-3, 2),
                                                                          ])

fh_mcmc_chainlux1, fhsampleslux1, fhstatslux1 = ahmc_bayesian_pinn_ode(prob, chainlux1,
                                                                       dataset = dataset,
                                                                       draw_samples = 1000,
                                                                       l2std = [0.05],
                                                                       phystd = [0.05],
                                                                       priorsNNw = (0.0,
                                                                                    3.0))

fh_mcmc_chainlux2, fhsampleslux2, fhstatslux2 = ahmc_bayesian_pinn_ode(prob, chainlux1,
                                                                       dataset = dataset,
                                                                       draw_samples = 1000,
                                                                       l2std = [0.05],
                                                                       phystd = [0.05],
                                                                       priorsNNw = (0.0,
                                                                                    3.0),
                                                                       param = [
                                                                           Normal(6.5, 2),
                                                                           Normal(-3, 2),
                                                                       ])

init1, re1 = destructure(chainflux1)
θinit, st = Lux.setup(Random.default_rng(), chainlux1)

#   PLOT testing points
t = sol.t
p = prob.p
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]

# Mean of last 500 sampled parameter's curves(flux chains)[Ensemble predictions]
out = re1.([fhsamplesflux1[i][1:22] for i in 500:1000])
yu = [out[i](t') for i in eachindex(out)]
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1_1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

@test mean(abs2.(sol.u .- meanscurve1_1)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve1_1)) < 4e-5

out = re1.([fhsamplesflux2[i][1:46] for i in 500:1000])
yu = [out[i](t') for i in eachindex(out)]
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1_2 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

@test mean(abs2.(sol.u .- meanscurve1_2)) < 1.5e-2
@test mean(abs2.(physsol1 .- meanscurve1_2)) < 1.5e-2

# estimated parameters(flux chain)
@test abs(fhsamplesflux2[1000][47] - p[1]) < abs(0.1 * p[1])
@test abs(fhsamplesflux2[1000][48] - p[2]) < abs(0.3 * p[2])

# Mean of last 500 sampled parameter's curves(lux chains)[Ensemble predictions]
θ = [vector_to_parameters(fhsampleslux1[i], θinit) for i in 500:1000]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(sol.u .- meanscurve2_1)) < 1.5e-5
@test mean(abs2.(physsol1 .- meanscurve2_1)) < 1.5e-5

θ = [vector_to_parameters(fhsampleslux2[i][1:(end - 2)], θinit) for i in 500:1000]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(sol.u .- meanscurve2_2)) < 3e-5
@test mean(abs2.(physsol1 .- meanscurve2_2)) < 3e-5

# estimated parameters(lux chain)
@test abs(fhsampleslux2[1000][47] - p[1]) < abs(0.1 * p[1])
@test abs(fhsampleslux2[1000][48] - p[2]) < abs(0.1 * p[2])