# Testing Code
using MCMCChains, ForwardDiff, Distributions, OrdinaryDiffEq
using NeuralPDE, Flux, OptimizationOptimisers, AdvancedHMC, Lux
using Statistics, Random, Functors, ComponentArrays, Test

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
                                                              draw_samples = 2500,
                                                              n_leapfrog = 30)
fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux,
                                                              dataset = dataset,
                                                              draw_samples = 2500,
                                                              n_leapfrog = 30)

init1, re1 = destructure(chainflux)
θinit, st = Lux.setup(Random.default_rng(), chainlux)

# TESTING TIMEPOINTS TO PLOT ON,Actual Sols and actual data
t = time
p = prob.p
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]

# Mean of last 1000 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
out = re1.(fhsamples1[(end - 500):end])
yu = collect(out[i](t') for i in eachindex(out))
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

θ = [vector_to_parameters(fhsamples2[i], θinit) for i in 2000:2500]
luxar = [chainlux(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(x̂ .- meanscurve1)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve1)) < 1e-5
@test mean(abs2.(x̂ .- meanscurve2)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve2)) < 1e-5

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
                                                              draw_samples = 2500,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 0.5)],
                                                              Metric = DiagEuclideanMetric,
                                                              n_leapfrog = 30)

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux1,
                                                              dataset = dataset,
                                                              draw_samples = 2500,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 0.5)],
                                                              Metric = DiagEuclideanMetric,
                                                              n_leapfrog = 30)

init1, re1 = destructure(chainflux1)
θinit, st = Lux.setup(Random.default_rng(), chainlux1)

# PLOT testing points
t = time
p = prob.p
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]

# Mean of last 1000 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
out = re1.([fhsamples1[i][1:22] for i in 2000:2500])
yu = collect(out[i](t') for i in eachindex(out))
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

θ = [vector_to_parameters(fhsamples2[i][1:(end - 1)], θinit) for i in 2000:2500]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(x̂ .- meanscurve1)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve1)) < 5e-5
@test mean(abs2.(x̂ .- meanscurve2)) < 5e-4
@test mean(abs2.(physsol1 .- meanscurve2)) < 5e-5

# ESTIMATED ODE PARAMETERS (NN1 AND NN2)
@test abs(p - mean([fhsamples2[i][23] for i in 2000:2500])) < 0.05
@test abs(p - mean([fhsamples1[i][23] for i in 2000:2500])) < 0.05

## PROBLEM-2
linear = (u, p, t) -> -u / p[1] + exp(t / p[2]) * cos(t)
tspan = (0.0, 10.0)
u0 = 0.0
p = [5.0, -5.0]
prob = ODEProblem(linear, u0, tspan, p)

# PROBLEM-2
linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))

# PLOT SOLUTION AND CREATE DATASET
sol = solve(prob, Tsit5(); saveat = 0.05)
u = sol.u[1:100]
time = sol.t[1:100]

# dataset and BPINN create
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
dataset = [x̂, time]

chainflux12 = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh), Flux.Dense(6, 1))
chainlux12 = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))

fh_mcmc_chainflux12, fhsamplesflux12, fhstatsflux12 = ahmc_bayesian_pinn_ode(prob,
                                                                             chainflux12,
                                                                             dataset = dataset,
                                                                             draw_samples = 1500,
                                                                             l2std = [0.05],
                                                                             phystd = [
                                                                                 0.05,
                                                                             ],
                                                                             priorsNNw = (0.0,
                                                                                          3.0),
                                                                             n_leapfrog = 30)

fh_mcmc_chainflux22, fhsamplesflux22, fhstatsflux22 = ahmc_bayesian_pinn_ode(prob,
                                                                             chainflux12,
                                                                             dataset = dataset,
                                                                             draw_samples = 1500,
                                                                             l2std = [0.05],
                                                                             phystd = [
                                                                                 0.05,
                                                                             ],
                                                                             priorsNNw = (0.0,
                                                                                          3.0),
                                                                             param = [
                                                                                 Normal(6.5,
                                                                                        0.5),
                                                                                 Normal(-3,
                                                                                        0.5),
                                                                             ],
                                                                             n_leapfrog = 30)

fh_mcmc_chainlux12, fhsampleslux12, fhstatslux12 = ahmc_bayesian_pinn_ode(prob, chainlux12,
                                                                          dataset = dataset,
                                                                          draw_samples = 1500,
                                                                          l2std = [0.05],
                                                                          phystd = [0.05],
                                                                          priorsNNw = (0.0,
                                                                                       3.0),
                                                                          n_leapfrog = 30)

fh_mcmc_chainlux22, fhsampleslux22, fhstatslux22 = ahmc_bayesian_pinn_ode(prob, chainlux12,
                                                                          dataset = dataset,
                                                                          draw_samples = 1500,
                                                                          l2std = [0.05],
                                                                          phystd = [0.05],
                                                                          priorsNNw = (0.0,
                                                                                       3.0),
                                                                          param = [
                                                                              Normal(6.5,
                                                                                     0.5),
                                                                              Normal(-3,
                                                                                     0.5),
                                                                          ],
                                                                          n_leapfrog = 30)

init1, re1 = destructure(chainflux12)
θinit, st = Lux.setup(Random.default_rng(), chainlux12)

#   PLOT testing points
t = sol.t
p = prob.p
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]

# Mean of last 500 sampled parameter's curves(flux chains)[Ensemble predictions]
out = re1.([fhsamplesflux12[i][1:61] for i in 1000:1500])
yu = [out[i](t') for i in eachindex(out)]
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1_1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

@test mean(abs2.(sol.u .- meanscurve1_1)) < 1e-4
@test mean(abs2.(physsol1 .- meanscurve1_1)) < 1e-4

out = re1.([fhsamplesflux22[i][1:61] for i in 1000:1500])
yu = [out[i](t') for i in eachindex(out)]
fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
meanscurve1_2 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean

@test mean(abs2.(sol.u .- meanscurve1_2)) < 5e-3
@test mean(abs2.(physsol1 .- meanscurve1_2)) < 5e-3

# estimated parameters(flux chain)
param1 = mean(i[62] for i in fhsamplesflux22[1000:1500])
param2 = mean(i[63] for i in fhsamplesflux22[1000:1500])
@test abs(param1 - p[1]) < abs(0.3 * p[1])
@test abs(param2 - p[2]) < abs(0.3 * p[2])

# Mean of last 500 sampled parameter's curves(lux chains)[Ensemble predictions]
θ = [vector_to_parameters(fhsampleslux12[i], θinit) for i in 1000:1500]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(sol.u .- meanscurve2_1)) < 1e-2
@test mean(abs2.(physsol1 .- meanscurve2_1)) < 1e-2

θ = [vector_to_parameters(fhsampleslux22[i][1:(end - 2)], θinit) for i in 1000:1500]
luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

@test mean(abs2.(sol.u .- meanscurve2_2)) < 2e-2
@test mean(abs2.(physsol1 .- meanscurve2_2)) < 2e-2

# estimated parameters(lux chain)
param1 = mean(i[62] for i in fhsampleslux22[1000:1500])
param2 = mean(i[63] for i in fhsampleslux22[1000:1500])
@test abs(param1 - p[1]) < abs(0.35 * p[1])
@test abs(param2 - p[2]) < abs(0.35 * p[2])