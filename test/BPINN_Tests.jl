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
    draw_samples = 2500,
    n_leapfrog = 30)

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux,
    draw_samples = 2500,
    n_leapfrog = 30)

# can change training strategies by adding this to call (Quadratuer and GridTraining show good results but stochastics sampling techniques perform bad)
# strategy = QuadratureTraining(; quadrature_alg = QuadGKJL(),
#     reltol = 1e-6,
#     abstol = 1e-3, maxiters = 1000,
#     batch = 0)

alg = NeuralPDE.BNNODE(chainflux, draw_samples = 2500,
    n_leapfrog = 30)
sol1flux = solve(prob, alg)

alg = NeuralPDE.BNNODE(chainlux, draw_samples = 2500,
    n_leapfrog = 30)
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
ta = range(tspan[1], tspan[2], length = 25)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) .+ (0.2 .* Array(u) .* randn(size(u))))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# testing points for solve call(saveat=1/50.0 ∴ at t = collect(eltype(saveat), prob.tspan[1]:saveat:prob.tspan[2] internally estimates)
ta0 = range(tspan[1], tspan[2], length = 101)
u1 = [linear_analytic(u0, p, ti) for ti in ta0]
x̂1 = collect(Float64, Array(u1) + 0.2 * randn(size(u1)))
time1 = vec(collect(Float64, ta0))
physsol1_1 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

using Plots, StatsPlots
# plot(dataset[2], calderivatives(dataset)')
yu = collect(prob.tspan[1]:(1 / 50.0):prob.tspan[2])
plot(yu, [linear_analytic(u0, p, t) for t in yu])
chainflux1 = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1)) |> Flux.f64
chainlux1 = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
init1, re1 = destructure(chainflux1)
θinit, st = Lux.setup(Random.default_rng(), chainlux1)

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
    dataset = dataset,
    draw_samples = 2500,
    physdt = 1 / 50.0f0,
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(9,
            0.5),
    ],
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

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 10.0),
    l2std = [0.005], phystd = [0.01],
    param = [Normal(11, 6)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30)
# original paper (pure data 0 1)
sol1flux = solve(prob, alg)
sol1flux.estimated_ode_params
# pure data method 1 1
sol2flux = solve(prob, alg)
sol2flux.estimated_ode_params
# pure data method 1 0
sol3flux = solve(prob, alg)
sol3flux.estimated_ode_params
# deri collocation
sol4flux = solve(prob, alg)
sol4flux.estimated_ode_params
# collocation
sol5flux = solve(prob, alg)
sol5flux.estimated_ode_params
# collocation + L2Data loss(at 9,0.5 1,2 gives same)
sol6flux = solve(prob, alg)
sol6flux.estimated_ode_params
# 2500 iters
sol7flux = solve(prob, alg)
sol7flux.estimated_ode_params

plotly()
plot!(yu, sol1flux.ensemblesol[1])
plot!(yu, sol2flux.ensemblesol[1])
plot!(yu, sol3flux.ensemblesol[1])
plot!(yu, sol4flux.ensemblesol[1])
plot!(yu, sol5flux.ensemblesol[1])
plot!(yu, sol6flux.ensemblesol[1])

plot!(dataset[2], dataset[1])

# plot!(sol4flux.ensemblesol[1])
# plot!(sol5flux.ensemblesol[1])

sol2flux.estimated_ode_params

sol1flux.estimated_ode_params

sol3flux.estimated_ode_params

sol4flux.estimated_ode_params

sol5flux.estimated_ode_params

alg = NeuralPDE.BNNODE(chainlux1, dataset = dataset,
    draw_samples = 2500,
    physdt = 1 / 50.0f0,
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(9,
            0.5),
    ],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30)

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
@test abs(p - mean([fhsamples2[i][23] for i in 2000:2500])) < abs(0.25 * p)
@test abs(p - mean([fhsamples1[i][23] for i in 2000:2500])) < abs(0.25 * p)

#-------------------------- solve() call  
@test mean(abs.(physsol1_1 .- sol2flux.ensemblesol[1])) < 8e-2
@test mean(abs.(physsol1_1 .- sol2lux.ensemblesol[1])) < 8e-2

# ESTIMATED ODE PARAMETERS (NN1 AND NN2)
@test abs(p - sol1flux.estimated_ode_params[1]) < abs(0.15 * p)
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

using Flux
using Random

function derivatives(chainflux, dataset)
    loss(x, y) = Flux.mse(chainflux(x), y)
    optimizer = Flux.Optimise.ADAM(0.01)
    epochs = 2500
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(chainflux), [(dataset[2]', dataset[1]')], optimizer)
    end
    getgradient(chainflux, dataset)
end

function getgradient(chainflux, dataset)
    return (chainflux(dataset[end]' .+ sqrt(eps(eltype(Float64)))) .-
            chainflux(dataset[end]')) ./
           sqrt(eps(eltype(dataset[end][1])))
end

ans = derivatives(chainflux12, dataset)

init3, re = destructure(chainflux12)
init2 == init1
init3 == init2
plot!(dataset[end], ans')
plot!(dataset[end], chainflux12(dataset[end]')')

ars = getgradient(chainflux12, dataset)

plot!(dataset[end], ars')

fh_mcmc_chainflux12, fhsamplesflux12, fhstatsflux12 = ahmc_bayesian_pinn_ode(prob,
    chainflux12,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [
        0.03],
    priorsNNw = (0.0,
        10.0),
    n_leapfrog = 30)

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
    ],
    n_leapfrog = 30)

fh_mcmc_chainlux12, fhsampleslux12, fhstatslux12 = ahmc_bayesian_pinn_ode(prob, chainlux12,
    draw_samples = 1500,
    l2std = [0.03],
    phystd = [0.03],
    priorsNNw = (0.0,
        10.0),
    n_leapfrog = 30)

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
    ],
    n_leapfrog = 30)

alg1 = NeuralPDE.BNNODE(chainflux12,
    dataset = dataset,
    draw_samples = 500,
    l2std = [0.01],
    phystd = [
        0.03,
    ],
    priorsNNw = (0.0,
        10.0),
    param = [
        Normal(-7,
            4),
    ],
    n_leapfrog = 30, progress = true)

# original paper (pure data 0 1)
sol1flux_pestim = solve(prob, alg1)
sol1flux_pestim.estimated_ode_params
# pure data method 1 1
sol2flux_pestim = solve(prob, alg1)
sol2flux_pestim.estimated_ode_params
# pure data method 1 0
sol3flux_pestim = solve(prob, alg1)
sol3flux_pestim.estimated_ode_params
# deri collocation
sol4flux_pestim = solve(prob, alg1)
sol4flux_pestim.estimated_ode_params
# collocation
sol5flux_pestim = solve(prob, alg1)
sol5flux_pestim.estimated_ode_params
# collocation + L2Data loss(at 9,0.5 1,2 gives same)
sol6flux_pestim = solve(prob, alg1)
sol6flux_pestim.estimated_ode_params

using Plots, StatsPlots
ars = collect(prob.tspan[1]:(1 / 50.0):prob.tspan[2])
plot(time, u)
plot!(ars, sol1flux_pestim.ensemblesol[1])
plot!(ars, sol2flux_pestim.ensemblesol[1])
plot!(ars, sol3flux_pestim.ensemblesol[1])
plot!(ars, sol4flux_pestim.ensemblesol[1])
plot!(ars, sol5flux_pestim.ensemblesol[1])
plot!(ars, sol6flux_pestim.ensemblesol[1])

sol3flux_pestim.estimated_ode_params

sol4flux_pestim.estimated_ode_params

sol5flux_pestim.estimated_ode_params

sol6flux_pestim.estimated_ode_params

ars = collect(prob.tspan[1]:(1 / 50.0):prob.tspan[2])

init, re1 = destructure(chainflux12)
init
init1
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
    ],
    n_leapfrog = 30)

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

using Plots, StatsPlots
using NoiseRobustDifferentiation, Weave, DataInterpolations

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood
# # 25 points 
# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
#     draw_samples = 1500, physdt = 1 / 50.0f0, phystd = [0.01],
#     l2std = [0.01],
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux1 = solve(prob, alg)
# sol2flux1.estimated_ode_params[1] #6.41722 Particles{Float64, 1}, 6.02404 Particles{Float64, 1}
# sol2flux2 = solve(prob, alg)
# sol2flux2.estimated_ode_params[1] #6.42782 Particles{Float64, 1}, 6.07509 Particles{Float64, 1}
# sol2flux3 = solve(prob, alg)
# sol2flux3.estimated_ode_params[1] #6.42782 Particles{Float64, 1}, 6.00825 Particles{Float64, 1}

# # 50 points 
# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux11 = solve(prob, alg)
# sol2flux11.estimated_ode_params[1] #5.71268 Particles{Float64, 1}, 6.07242 Particles{Float64, 1}
# sol2flux22 = solve(prob, alg)
# sol2flux22.estimated_ode_params[1] #5.74599 Particles{Float64, 1}, 6.04837 Particles{Float64, 1}
# sol2flux33 = solve(prob, alg)
# sol2flux33.estimated_ode_params[1] #5.74599 Particles{Float64, 1}, 6.02838 Particles{Float64, 1}

# # 100 points 
# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux111 = solve(prob, alg)
# sol2flux111.estimated_ode_params[1] #6.59097 Particles{Float64, 1}, 5.89384 Particles{Float64, 1}
# sol2flux222 = solve(prob, alg)
# sol2flux222.estimated_ode_params[1] #6.62813 Particles{Float64, 1}, 5.88216 Particles{Float64, 1}
# sol2flux333 = solve(prob, alg)
# sol2flux333.estimated_ode_params[1] #6.62813 Particles{Float64, 1}, 5.85327 Particles{Float64, 1}

# # ----------------------------------------------------------
# # physdt=1/20, full likelihood cdm
# # 25 points 
# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux1_cdm = solve(prob, alg)
# sol2flux1_cdm.estimated_ode_params[1]# 6.50506 Particles{Float64, 1} ,6.38963 Particles{Float64, 1}
# sol2flux2_cdm = solve(prob, alg)
# sol2flux2_cdm.estimated_ode_params[1] #6.50032 Particles{Float64, 1} ,6.39817 Particles{Float64, 1}
# sol2flux3_cdm = solve(prob, alg)
# sol2flux3_cdm.estimated_ode_params[1] #6.50032 Particles{Float64, 1} ,6.36296 Particles{Float64, 1}

# # 50 points 
# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux11_cdm = solve(prob, alg)
# sol2flux11_cdm.estimated_ode_params[1] #6.52951 Particles{Float64, 1},5.15621 Particles{Float64, 1}
# sol2flux22_cdm = solve(prob, alg)
# sol2flux22_cdm.estimated_ode_params[1] #6.54988 Particles{Float64, 1},5.16363 Particles{Float64, 1}
# sol2flux33_cdm = solve(prob, alg)
# sol2flux33_cdm.estimated_ode_params[1] #6.54988 Particles{Float64, 1},5.15591 Particles{Float64, 1}

# # 100 points 
# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux111_cdm = solve(prob, alg)
# sol2flux111_cdm.estimated_ode_params[1] #6.74338 Particles{Float64, 1}, 9.72422 Particles{Float64, 1}
# sol2flux222_cdm = solve(prob, alg)
# sol2flux222_cdm.estimated_ode_params[1] #6.72642 Particles{Float64, 1}, 9.71991 Particles{Float64, 1}
# sol2flux333_cdm = solve(prob, alg)
# sol2flux333_cdm.estimated_ode_params[1] #6.72642 Particles{Float64, 1}, 9.75045 Particles{Float64, 1}

# --------------------------------------------------------------------------------------
#                              NEW SERIES OF TESTS (IN ORDER OF EXECUTION)
#  -------------------------------------------------------------------------------------
# original paper implementaion
# 25 points
ta = range(tspan[1], tspan[2], length = 25)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, u .+ 0.05 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset1 = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]
# scatter!(time, u)
# dataset
# scatter!(dataset1[2], dataset1[1])
# plot(time, physsol1)

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux1_normal = solve(prob, alg)
sol2flux1_normal.estimated_ode_params[1]  #7.70593 Particles{Float64, 1}, 6.36096 Particles{Float64, 1} | 6.45865 Particles{Float64, 1}
sol2flux2_normal = solve(prob, alg)
sol2flux2_normal.estimated_ode_params[1] #6.66347 Particles{Float64, 1}, 6.36974 Particles{Float64, 1} | 6.45865 Particles{Float64, 1}
sol2flux3_normal = solve(prob, alg)
sol2flux3_normal.estimated_ode_params[1] #6.84827 Particles{Float64, 1}, 6.29555 Particles{Float64, 1} | 6.39947 Particles{Float64, 1}

# 50 points
ta = range(tspan[1], tspan[2], length = 50)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset2 = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux11_normal = solve(prob, alg)
sol2flux11_normal.estimated_ode_params[1] #7.83577 Particles{Float64, 1},6.24652 Particles{Float64, 1} | 6.34495 Particles{Float64, 1}
sol2flux22_normal = solve(prob, alg)
sol2flux22_normal.estimated_ode_params[1] #6.49477 Particles{Float64, 1},6.2118 Particles{Float64, 1} | 6.32476 Particles{Float64, 1}
sol2flux33_normal = solve(prob, alg)
sol2flux33_normal.estimated_ode_params[1] #6.47421 Particles{Float64, 1},6.33687 Particles{Float64, 1} | 6.2448 Particles{Float64, 1}

# 100 points
ta = range(tspan[1], tspan[2], length = 100)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset3 = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux111_normal = solve(prob, alg)
sol2flux111_normal.estimated_ode_params[1] #5.96604 Particles{Float64, 1},5.99588 Particles{Float64, 1} | 6.19805 Particles{Float64, 1}
sol2flux222_normal = solve(prob, alg)
sol2flux222_normal.estimated_ode_params[1] #6.05432 Particles{Float64, 1},6.0768 Particles{Float64, 1} | 6.22948 Particles{Float64, 1}
sol2flux333_normal = solve(prob, alg)
sol2flux333_normal.estimated_ode_params[1] #6.08856 Particles{Float64, 1},5.94819 Particles{Float64, 1} | 6.2551 Particles{Float64, 1}

# LOTKA VOLTERRA CASE 
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
u01 = [1.0, 1.0]
p1 = [1.5, 1.0, 3.0, 1.0]
tspan1 = (0.0, 6.0)
prob1 = ODEProblem(lotka_volterra, u01, tspan1, p1)

# chainlux = Lux.Chain(Lux.Dense(1, 7, Lux.tanh), Lux.Dense(7, 7, Lux.tanh), Lux.Dense(7, 2))
chainflux1 = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 8, tanh), Flux.Dense(8, 2))

#testing timepoints must match keyword arg `saveat`` timepoints of solve() call
t1 = collect(Float64, prob1.tspan[1]:(1 / 50.0):prob1.tspan[2])

# --------------------------------------------------------------------------
# original paper implementaion lotka volterra
# 31 points  
solution1 = solve(prob1, Tsit5(); saveat = 0.1)
time1 = solution1.t
physsol1_1 = solution1.u
u1 = hcat(solution1.u...)
x1 = u1[1, :] .+ 0.3 .* u1[1, :] .* randn(length(u1[1, :]))
y1 = u1[2, :] .+ 0.3 .* u1[2, :] .* randn(length(u1[2, :]))
dataset2_1 = [x1, y1, time1]
plot(dataset2_1[end], dataset2_1[1])
plot!(dataset2_1[end], dataset2_1[2])
plot!(time1, u1[1, :])
plot!(time1, u1[2, :])

alg1 = NeuralPDE.BNNODE(chainflux1,
    dataset = dataset2_1,
    draw_samples = 1000,
    physdt = 1 / 20.0,
    l2std = [
        0.2,
        0.2,
    ],
    phystd = [
        0.5,
        0.5,
    ],
    priorsNNw = (0.0,
        10.0),
    param = [
        Normal(4,
            3),
        Normal(-2,
            4),
        Normal(0,
            5),
        Normal(2.5,
            2)],
    n_leapfrog = 30, progress = true)

# original paper (pure data 0 1)
sol1flux1_lotka = solve(prob1, alg1)
sol1flux1_lotka.estimated_ode_params
# pure data method 1 1
sol2flux1_lotka = solve(prob1, alg1)
sol2flux1_lotka.estimated_ode_params
# pure data method 1 0
sol3flux1_lotka = solve(prob1, alg1)
sol3flux1_lotka.estimated_ode_params
# deri collocation
sol4flux1_lotka = solve(prob1, alg1)
sol4flux1_lotka.estimated_ode_params
# collocation
sol5flux1_lotka = solve(prob1, alg1)
sol5flux1_lotka.estimated_ode_params
# collocation + L2Data loss(at 9,0.5 1,2 gives same)
sol6flux1_lotka = solve(prob1, alg1)
sol6flux1_lotka.estimated_ode_params

sol7flux1_lotka = solve(prob1, alg1)
sol7flux1_lotka.estimated_ode_params

using Plots, StatsPlots
plot(dataset2_1[3], u1[1, :])
plot!(dataset2_1[3], u1[2, :])
plot!(collect(prob1.tspan[1]:(1 / 50.0):prob1.tspan[2]), sol5flux1_normal.ensemblesol[2])
plot!(collect(prob1.tspan[1]:(1 / 50.0):prob1.tspan[2]),
    sol1flux1_normal.ensemblesol[1],
    legend = :outerbottomleft)
sol1flux2_normal = solve(prob1, alg1)
sol1flux2_normal.estimated_ode_params  #|
sol1flux3_normal = solve(prob1, alg1)
sol1flux3_normal.estimated_ode_params  #|
sol1flux4_normal = solve(prob1, alg1)
sol1flux4_normal.estimated_ode_params

plotly()
plot!(title = "yuh")
plot!(dataset2_1[3], dataset2_1[1])
plot!(collect(prob1.tspan[1]:(1 / 50.0):prob1.tspan[2]), sol1flux1_normal.ensemblesol[1])
plot!(collect(prob1.tspan[1]:(1 / 50.0):prob1.tspan[2]), sol1flux2_normal.ensemblesol[1])
plot!(collect(prob1.tspan[1]:(1 / 50.0):prob1.tspan[2]), sol1flux3_normal.ensemblesol[2])
plot!(collect(prob1.tspan[1]:(1 / 50.0):prob1.tspan[2]), sol1flux4_normal.ensemblesol[1])
plot(time1, u1[1, :])
plot!(time1, u1[2, :])

ars = chainflux1(dataset2_1[end]')
plot(ars[1, :])
plot!(ars[2, :])

function calculate_derivatives(dataset)
    u = dataset[1]
    u1 = dataset[2]
    t = dataset[end]
    # control points
    n = Int(floor(length(t) / 10))
    # spline for datasetvalues(solution) 
    # interp = BSplineApprox(u, t, 4, 10, :Uniform, :Uniform)
    interp = CubicSpline(u, t)
    interp1 = CubicSpline(u1, t)
    # derrivatives interpolation
    dx = t[2] - t[1]
    time = collect(t[1]:dx:t[end])
    smoothu = [interp(i) for i in time]
    smoothu1 = [interp1(i) for i in time]
    # derivative of the spline (must match function derivative) 
    û = tvdiff(smoothu, 20, 0.5, dx = dx, ε = 1)
    û1 = tvdiff(smoothu1, 20, 0.5, dx = dx, ε = 1)
    # tvdiff(smoothu, 100, 0.035, dx = dx, ε = 1)
    # FDM
    # û1 = diff(u) / dx
    # dataset[1] and smoothu are almost equal(rounding errors)
    return û, û1
    # return 1
end

ar = calculate_derivatives(dataset2_1)
plot(ar[1])
plot!(ar[2])

# 61 points
solution1 = solve(prob1, Tsit5(); saveat = 0.1)
time1 = solution1.t
physsol1_1 = solution1.u
u1 = hcat(solution1.u...)
x1 = u1[1, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
y1 = u1[2, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
dataset2_2 = [x1, y1, time1]

alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_2,
    draw_samples = 1000,
    l2std = [
        0.1,
        0.1,
    ],
    phystd = [
        0.1,
        0.1,
    ],
    priorsNNw = (0.0,
        5.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux11_normal = solve(prob1, alg1)
sol1flux11_normal.estimated_ode_params #|
sol1flux22_normal = solve(prob1, alg1)
sol1flux22_normal.estimated_ode_params #|
sol1flux33_normal = solve(prob1, alg1)
sol1flux33_normal.estimated_ode_params #|

# 121 points
solution1 = solve(prob1, Tsit5(); saveat = 0.05)
time1 = solution1.t
physsol1_1 = solution1.u
u1 = hcat(solution1.u...)
x1 = u1[1, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
y1 = u1[2, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
dataset2_3 = [x1, y1, time1]

alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_3,
    draw_samples = 1000,
    l2std = [
        0.1,
        0.1,
    ],
    phystd = [
        0.1,
        0.1,
    ],
    priorsNNw = (0.0,
        5.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux111_normal = solve(prob1, alg1)
sol1flux111_normal.estimated_ode_params #|
sol1flux222_normal = solve(prob1, alg1)
sol1flux222_normal.estimated_ode_params #|
sol1flux333_normal = solve(prob1, alg1)
sol1flux333_normal.estimated_ode_params #| 

# -------------------------------------------------------------------- 

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -640.4155412187399
# Sampling 100%|███████████████████████████████| Time: 0:02:30

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -640.4155412187399
# Sampling 100%|███████████████████████████████| Time: 0:01:54

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -640.4155412187399
# Sampling 100%|███████████████████████████████| Time: 0:01:59

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1198.9147562830894
# Sampling 100%|███████████████████████████████| Time: 0:02:44

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1198.9147562830894
# Sampling 100%|███████████████████████████████| Time: 0:02:41

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1198.9147562830894
# Sampling 100%|███████████████████████████████| Time: 0:02:41

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -2473.741390504424
# Sampling 100%|███████████████████████████████| Time: 0:03:52

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -2473.741390504424
# Sampling 100%|███████████████████████████████| Time: 0:03:49

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -2473.741390504424
# Sampling 100%|███████████████████████████████| Time: 0:03:50

# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# physics Logpdf is : -6.659143464386241e7
# prior Logpdf is : -150.30074579848434
# L2lossData Logpdf is : -6.03075717462954e6
# Sampling 100%|███████████████████████████████| Time: 0:04:54

# physics Logpdf is : -8.70012053004202e8
# prior Logpdf is : -150.3750892952511
# L2lossData Logpdf is : -6.967914805207133e6
# Sampling 100%|███████████████████████████████| Time: 0:05:09

# physics Logpdf is : -5.417241281343099e7
# prior Logpdf is : -150.52079555737976
# L2lossData Logpdf is : -4.195953436792884e6
# Sampling 100%|███████████████████████████████| Time: 0:05:01

# physics Logpdf is : -4.579552981943833e8
# prior Logpdf is : -150.30491731974283
# L2lossData Logpdf is : -8.595475827260146e6
# Sampling 100%|███████████████████████████████| Time: 0:06:08

# physics Logpdf is : -1.989281834955769e7
# prior Logpdf is : -150.16009042727543
# L2lossData Logpdf is : -1.121270659669029e7
# Sampling 100%|███████████████████████████████| Time: 0:05:38

# physics Logpdf is : -8.683829147264534e8
# prior Logpdf is : -150.37824872259102
# L2lossData Logpdf is : -1.0887662888035845e7
# Sampling 100%|███████████████████████████████| Time: 0:05:50

# physics Logpdf is : -3.1944760610332566e8
# prior Logpdf is : -150.33610348737565
# L2lossData Logpdf is : -1.215458786744478e7
# Sampling 100%|███████████████████████████████| Time: 0:10:50

# physics Logpdf is : -3.2884572300341567e6
# prior Logpdf is : -150.21002268156343
# L2lossData Logpdf is : -1.102536731511176e7
# Sampling 100%|███████████████████████████████| Time: 0:09:53

# physics Logpdf is : -5.31293521002414e8
# prior Logpdf is : -150.20948536040126
# L2lossData Logpdf is : -1.818717239584132e7
# Sampling 100%|███████████████████████████████| Time: 0:08:53

# ----------------------------------------------------------
# Full likelihood no l2 only new L22(NN gradients)
# 25 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux1_new = solve(prob, alg)
sol2flux1_new.estimated_ode_params[1] #5.35705 Particles{Float64, 1},5.91809 Particles{Float64, 1} | 6.21662 Particles{Float64, 1}
sol2flux2_new = solve(prob, alg)
sol2flux2_new.estimated_ode_params[1] #6.73629 Particles{Float64, 1},5.966 Particles{Float64, 1} | 7.14238 Particles{Float64, 1}
sol2flux3_new = solve(prob, alg)
sol2flux3_new.estimated_ode_params[1] #4.64324 Particles{Float64, 1},5.9559 Particles{Float64, 1} | 6.79159 Particles{Float64, 1}

# 50 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux11_new = solve(prob, alg)
sol2flux11_new.estimated_ode_params[1] #6.43659 Particles{Float64, 1},6.03723 Particles{Float64, 1} | 5.33467 Particles{Float64, 1}
sol2flux22_new = solve(prob, alg)
sol2flux22_new.estimated_ode_params[1] # 6.4389 Particles{Float64, 1},6.01308 Particles{Float64, 1} | 6.52419 Particles{Float64, 1}
sol2flux33_new = solve(prob, alg)
sol2flux33_new.estimated_ode_params[1] # 7.10082 Particles{Float64, 1}, 6.03989 Particles{Float64, 1} | 5.36921 Particles{Float64, 1}

# 100 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux111_new = solve(prob, alg)
sol2flux111_new.estimated_ode_params[1] #6.94385 Particles{Float64, 1},5.87832 Particles{Float64, 1} | 6.45333 Particles{Float64, 1}
sol2flux222_new = solve(prob, alg)
sol2flux222_new.estimated_ode_params[1] #5.888 Particles{Float64, 1},5.86901 Particles{Float64, 1} | 4.64417 Particles{Float64, 1}
sol2flux333_new = solve(prob, alg)
sol2flux333_new.estimated_ode_params[1] #6.96835 Particles{Float64, 1},5.86708 Particles{Float64, 1} | 5.88037 Particles{Float64, 1}
# ---------------------------------------------------------------------------

# ----------------------------------------------------------
# Full likelihood  l2 + new L22(NN gradients)
# 25 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux1_new_all = solve(prob, alg)
sol2flux1_new_all.estimated_ode_params[1] #5.35705 Particles{Float64, 1},5.91809 Particles{Float64, 1} | 6.4358 Particles{Float64, 1}
sol2flux2_new_all = solve(prob, alg)
sol2flux2_new_all.estimated_ode_params[1] #6.73629 Particles{Float64, 1},5.966 Particles{Float64, 1} | 6.52449 Particles{Float64, 1}
sol2flux3_new_all = solve(prob, alg)
sol2flux3_new_all.estimated_ode_params[1] #4.64324 Particles{Float64, 1},5.9559 Particles{Float64, 1} | 6.34188 Particles{Float64, 1}

# 50 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux11_new_all = solve(prob, alg)
sol2flux11_new_all.estimated_ode_params[1] #6.43659 Particles{Float64, 1},6.03723 Particles{Float64, 1} | 6.37889 Particles{Float64, 1}
sol2flux22_new_all = solve(prob, alg)
sol2flux22_new_all.estimated_ode_params[1] # 6.4389 Particles{Float64, 1},6.01308 Particles{Float64, 1} | 6.34747 Particles{Float64, 1}
sol2flux33_new_all = solve(prob, alg)
sol2flux33_new_all.estimated_ode_params[1] # 7.10082 Particles{Float64, 1}, 6.03989 Particles{Float64, 1} | 6.39699 Particles{Float64, 1}

# 100 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux111_new_all = solve(prob, alg)
sol2flux111_new_all.estimated_ode_params[1] #6.94385 Particles{Float64, 1},5.87832 Particles{Float64, 1} | 6.24327 Particles{Float64, 1}
sol2flux222_new_all = solve(prob, alg)
sol2flux222_new_all.estimated_ode_params[1] #5.888 Particles{Float64, 1},5.86901 Particles{Float64, 1} | 6.23928 Particles{Float64, 1}
sol2flux333_new_all = solve(prob, alg)
sol2flux333_new_all.estimated_ode_params[1] #6.96835 Particles{Float64, 1},5.86708 Particles{Float64, 1} | 6.2145 Particles{Float64, 1}

# ---------------------------------------------------------------------------
# Full likelihood  l2 + new L22(dataset gradients) lotka volterra
# 36 points 
alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_1,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux1_new_all = solve(prob1, alg1)
sol1flux1_new_all.estimated_ode_params[1]  #|
sol1flux2_new_all = solve(prob1, alg1)
sol1flux2_new_all.estimated_ode_params[1] #|
sol1flux3_new_all = solve(prob1, alg1)
sol1flux3_new_all.estimated_ode_params[1] #|

# 61 points 
alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_2,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux11_new_all = solve(prob1, alg1)
sol1flux11_new_all.estimated_ode_params[1] #|
sol1flux22_new_all = solve(prob1, alg1)
sol1flux22_new_all.estimated_ode_params[1] #|
sol1flux33_new_all = solve(prob1, alg1)
sol1flux33_new_all.estimated_ode_params[1] #|

# 121 points 
alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_3,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux111_new_all = solve(prob1, alg1)
sol1flux111_new_all.estimated_ode_params[1] #|
sol1flux222_new_all = solve(prob1, alg1)
sol1flux222_new_all.estimated_ode_params[1] #|
sol1flux333_new_all = solve(prob1, alg1)
sol1flux333_new_all.estimated_ode_params[1] #|
# -------------------------------------------------------------------- 

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -640.4155412187399
# L2loss2 Logpdf is : -757.9047847584478
# Sampling 100%|███████████████████████████████| Time: 0:02:32

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -640.4155412187399
# L2loss2 Logpdf is : -757.9047847584478
# Sampling 100%|███████████████████████████████| Time: 0:02:19

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -640.4155412187399
# L2loss2 Logpdf is : -757.9047847584478
# Sampling 100%|███████████████████████████████| Time: 0:02:31

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1198.9147562830894
# L2loss2 Logpdf is : -1517.3653615845183
# Sampling 100%|███████████████████████████████| Time: 0:03:45

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1198.9147562830894
# L2loss2 Logpdf is : -1517.3653615845183
# Sampling 100%|███████████████████████████████| Time: 0:03:20

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1198.9147562830894
# L2loss2 Logpdf is : -1517.3653615845183
# Sampling 100%|███████████████████████████████| Time: 0:03:20

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -2473.741390504424
# L2loss2 Logpdf is : -3037.8868319811254
# Sampling 100%|███████████████████████████████| Time: 0:04:57

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -2473.741390504424
# L2loss2 Logpdf is : -3037.8868319811254
# Sampling 100%|███████████████████████████████| Time: 0:05:26

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -2473.741390504424
# L2loss2 Logpdf is : -3037.8868319811254
# Sampling 100%|███████████████████████████████| Time: 0:05:01

# ----------------------------------------------------------
# Full likelihood  l2 + new L22(dataset gradients)
# 25 points
# 1*,2*,  
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux1_newdata_all = solve(prob, alg)
sol2flux1_newdata_all.estimated_ode_params[1] #5.35705 Particles{Float64, 1},5.91809 Particles{Float64, 1} | 5.73072 Particles{Float64, 1}
sol2flux2_newdata_all = solve(prob, alg)
sol2flux2_newdata_all.estimated_ode_params[1] #6.73629 Particles{Float64, 1},5.966 Particles{Float64, 1} | 5.71597 Particles{Float64, 1}
sol2flux3_newdata_all = solve(prob, alg)
sol2flux3_newdata_all.estimated_ode_params[1] #4.64324 Particles{Float64, 1},5.9559 Particles{Float64, 1} | 5.7313 Particles{Float64, 1}

# 50 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux11_newdata_all = solve(prob, alg)
sol2flux11_newdata_all.estimated_ode_params[1] #6.43659 Particles{Float64, 1},6.03723 Particles{Float64, 1} | 6.07153 Particles{Float64, 1}
sol2flux22_newdata_all = solve(prob, alg)
sol2flux22_newdata_all.estimated_ode_params[1] # 6.4389 Particles{Float64, 1},6.01308 Particles{Float64, 1} | 6.06623 Particles{Float64, 1}
sol2flux33_newdata_all = solve(prob, alg)
sol2flux33_newdata_all.estimated_ode_params[1] # 7.10082 Particles{Float64, 1}, 6.03989 Particles{Float64, 1} | 6.12748 Particles{Float64, 1}

# 100 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux111_newdata_all = solve(prob, alg)
sol2flux111_newdata_all.estimated_ode_params[1] #6.94385 Particles{Float64, 1},5.87832 Particles{Float64, 1} | 6.26222 Particles{Float64, 1}
sol2flux222_newdata_all = solve(prob, alg)
sol2flux222_newdata_all.estimated_ode_params[1] #5.888 Particles{Float64, 1},5.86901 Particles{Float64, 1} | 5.86494 Particles{Float64, 1}
sol2flux333_newdata_all = solve(prob, alg)
sol2flux333_newdata_all.estimated_ode_params[1] #6.96835 Particles{Float64, 1},5.86708 Particles{Float64, 1} |  

# ---------------------------------------------------------------------------

# LOTKA VOLTERRA CASE
using Plots, StatsPlots
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
u01 = [1.0, 1.0]
p1 = [1.5, 1.0, 3.0, 1.0]
tspan1 = (0.0, 6.0)
prob1 = ODEProblem(lotka_volterra, u01, tspan1, p1)

chainlux = Lux.Chain(Lux.Dense(1, 6, Lux.tanh), Lux.Dense(6, 6, Lux.tanh), Lux.Dense(6, 2))

#testing timepoints must match keyword arg `saveat`` timepoints of solve() call
t1 = collect(Float64, prob.tspan[1]:(1 / 50.0):prob.tspan[2])

# --------------------------------------------------------------------------
# original paper implementaion
# 25 points  
solution1 = solve(prob1, Tsit5(); saveat = 0.2)
time1 = solution1.t
physsol1_1 = solution1.u
u1 = hcat(solution1.u...)
x1 = u1[1, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
y1 = u1[2, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
dataset2_1 = [x1, y1, time1]

plot(time1, u1[1, :])
plot!(time1, u1[2, :])
scatter!(dataset2_1[3], dataset2_1[1])
scatter!(dataset2_1[3], dataset2_1[2])

alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_1,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux1_normal = solve(prob1, alg1)
sol1flux1_normal.estimated_ode_params[1]  #|
sol1flux2_normal = solve(prob1, alg1)
sol1flux2_normal.estimated_ode_params[1] #|
sol1flux3_normal = solve(prob1, alg1)
sol1flux3_normal.estimated_ode_params[1] #|

# 50 points
solution1 = solve(prob1, Tsit5(); saveat = 0.05)
time1 = solution1.t
physsol1_1 = solution1.u
u1 = hcat(solution1.u...)
x1 = u1[1, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
y1 = u1[2, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
dataset2_2 = [x1, y1, time1]

alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_2,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux11_normal = solve(prob1, alg1)
sol1flux11_normal.estimated_ode_params[1] #|
sol1flux22_normal = solve(prob1, alg1)
sol1flux22_normal.estimated_ode_params[1] #|
sol1flux33_normal = solve(prob1, alg1)
sol1flux33_normal.estimated_ode_params[1] #|

# 100 points
solution = solve(prob1, Tsit5(); saveat = 0.05)
time1 = solution1.t
physsol1_1 = solution1.u
u1 = hcat(solution1.u...)
x1 = u1[1, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
y1 = u1[2, :] + 0.4 .* u1[1, :] .* randn(length(u1[1, :]))
dataset2_3 = [x1, y1, time1]

alg1 = NeuralPDE.BNNODE(chainlux,
    dataset = dataset2_3,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol1flux111_normal = solve(prob1, alg1)
sol1flux111_normal.estimated_ode_params[1] #|
sol1flux222_normal = solve(prob1, alg1)
sol1flux222_normal.estimated_ode_params[1] #|
sol1flux333_normal = solve(prob1, alg1)
sol1flux333_normal.estimated_ode_params[1] #|

# --------------------------------------------------------------------

# ----------------------------------------------------------
# Full likelihood no l2 only new L22(NN gradients)
# 25 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux1_new = solve(prob, alg)
sol2flux1_new.estimated_ode_params[1] #5.35705 Particles{Float64, 1},5.91809 Particles{Float64, 1} |
sol2flux2_new = solve(prob, alg)
sol2flux2_new.estimated_ode_params[1] #6.73629 Particles{Float64, 1},5.966 Particles{Float64, 1}   |
sol2flux3_new = solve(prob, alg)
sol2flux3_new.estimated_ode_params[1] #4.64324 Particles{Float64, 1},5.9559 Particles{Float64, 1}  |

# 50 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux11_new = solve(prob, alg)
sol2flux11_new.estimated_ode_params[1] #6.43659 Particles{Float64, 1},6.03723 Particles{Float64, 1}   |
sol2flux22_new = solve(prob, alg)
sol2flux22_new.estimated_ode_params[1] # 6.4389 Particles{Float64, 1},6.01308 Particles{Float64, 1}   |
sol2flux33_new = solve(prob, alg)
sol2flux33_new.estimated_ode_params[1] # 7.10082 Particles{Float64, 1}, 6.03989 Particles{Float64, 1} |

# 100 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux111_new = solve(prob, alg)
sol2flux111_new.estimated_ode_params[1] #6.94385 Particles{Float64, 1},5.87832 Particles{Float64, 1}   |
sol2flux222_new = solve(prob, alg)
sol2flux222_new.estimated_ode_params[1] #5.888 Particles{Float64, 1},5.86901 Particles{Float64, 1}     |
sol2flux333_new = solve(prob, alg)
sol2flux333_new.estimated_ode_params[1] #6.96835 Particles{Float64, 1},5.86708 Particles{Float64, 1}   |
# ---------------------------------------------------------------------------

# ----------------------------------------------------------
# Full likelihood  l2 + new L22(NN gradients)
# 25 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux1_new_all = solve(prob, alg)
sol2flux1_new_all.estimated_ode_params[1] #5.35705 Particles{Float64, 1},5.91809 Particles{Float64, 1}  |
sol2flux2_new_all = solve(prob, alg)
sol2flux2_new_all.estimated_ode_params[1] #6.73629 Particles{Float64, 1},5.966 Particles{Float64, 1}    |
sol2flux3_new_all = solve(prob, alg)
sol2flux3_new_all.estimated_ode_params[1] #4.64324 Particles{Float64, 1},5.9559 Particles{Float64, 1}   |

# 50 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux11_new_all = solve(prob, alg)
sol2flux11_new_all.estimated_ode_params[1] #6.43659 Particles{Float64, 1},6.03723 Particles{Float64, 1}   |
sol2flux22_new_all = solve(prob, alg)
sol2flux22_new_all.estimated_ode_params[1] # 6.4389 Particles{Float64, 1},6.01308 Particles{Float64, 1}   |
sol2flux33_new_all = solve(prob, alg)
sol2flux33_new_all.estimated_ode_params[1] # 7.10082 Particles{Float64, 1}, 6.03989 Particles{Float64, 1} |
# 100 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol2flux111_new_all = solve(prob, alg)
sol2flux111_new_all.estimated_ode_params[1] #6.94385 Particles{Float64, 1},5.87832 Particles{Float64, 1}  |
sol2flux222_new_all = solve(prob, alg)
sol2flux222_new_all.estimated_ode_params[1] #5.888 Particles{Float64, 1},5.86901 Particles{Float64, 1}    |
sol2flux333_new_all = solve(prob, alg)
sol2flux333_new_all.estimated_ode_params[1] #6.96835 Particles{Float64, 1},5.86708 Particles{Float64, 1}  |

# ---------------------------------------------------------------------------

# ----------------------------------------------------------
# Full likelihood  l2 + new L22(dataset gradients)
# 25 points 
# *1,*2 vs *2.5
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset1,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol1flux1_newdata_all = solve(prob, alg)
sol1flux1_newdata_all.estimated_ode_params[1] #5.35705 Particles{Float64, 1},5.91809 Particles{Float64, 1} |
sol1flux2_newdata_all = solve(prob, alg)
sol1flux2_newdata_all.estimated_ode_params[1] #6.73629 Particles{Float64, 1},5.966 Particles{Float64, 1}   |
sol1flux3_newdata_all = solve(prob, alg)
sol1flux3_newdata_all.estimated_ode_params[1] #4.64324 Particles{Float64, 1},5.9559 Particles{Float64, 1}  |

# 50 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset2,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol1flux11_newdata_all = solve(prob, alg)
sol1flux11_newdata_all.estimated_ode_params[1] #6.43659 Particles{Float64, 1},6.03723 Particles{Float64, 1}    |
sol1flux22_newdata_all = solve(prob, alg)
sol1flux22_newdata_all.estimated_ode_params[1] # 6.4389 Particles{Float64, 1},6.01308 Particles{Float64, 1}    |
sol1flux33_newdata_all = solve(prob, alg)
sol1flux33_newdata_all.estimated_ode_params[1] # 7.10082 Particles{Float64, 1}, 6.03989 Particles{Float64, 1}   |

# 100 points 
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset3,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

sol1flux111_newdata_all = solve(prob, alg)
sol1flux111_newdata_all.estimated_ode_params[1]  #|
sol1flux222_newdata_all = solve(prob, alg)
sol1flux222_newdata_all.estimated_ode_params[1]  #|
sol1flux333_newdata_all = solve(prob, alg)
sol1flux333_newdata_all.estimated_ode_params[1] #6.96835 Particles{Float64, 1},5.86708 Particles{Float64, 1}   |

# ------------------------------------------------------------------------------------------------------------------------------

# sol2flux111.estimated_ode_params[1]
# # mine *5
# 7.03386Particles{Float64, 1}
# # normal
# 6.38951Particles{Float64, 1}
# 6.67657Particles{Float64, 1}
# # mine *10
# 7.53672Particles{Float64, 1}
# # mine *2
# 6.29005Particles{Float64, 1}
# 6.29844Particles{Float64, 1}

# # new mine *2
# 6.39008Particles{Float64, 1}
# 6.22071Particles{Float64, 1}
# 6.15611Particles{Float64, 1}

# # new mine *2 tvdiff(smoothu, 20, 0.035, dx = dx, ε = 1e-2)
# 6.25549Particles{Float64, 1}
# ----------------------------------------------------------

# ---------------------------------------------------

function calculate_derivatives1(dataset)
    x̂, time = dataset
    num_points = length(x̂)
    # Initialize an array to store the derivative values.
    derivatives = similar(x̂)

    for i in 2:(num_points - 1)
        # Calculate the first-order derivative using central differences.
        Δt_forward = time[i + 1] - time[i]
        Δt_backward = time[i] - time[i - 1]

        derivative = (x̂[i + 1] - x̂[i - 1]) / (Δt_forward + Δt_backward)

        derivatives[i] = derivative
    end

    # Derivatives at the endpoints can be calculated using forward or backward differences.
    derivatives[1] = (x̂[2] - x̂[1]) / (time[2] - time[1])
    derivatives[end] = (x̂[end] - x̂[end - 1]) / (time[end] - time[end - 1])
    return derivatives
end

function calculate_derivatives2(dataset)
    u = dataset[1]
    t = dataset[2]
    # control points
    n = Int(floor(length(t) / 10))
    # spline for datasetvalues(solution) 
    # interp = BSplineApprox(u, t, 4, 10, :Uniform, :Uniform)
    interp = CubicSpline(u, t)
    # derrivatives interpolation
    dx = t[2] - t[1]
    time = collect(t[1]:dx:t[end])
    smoothu = [interp(i) for i in time]
    # derivative of the spline (must match function derivative) 
    û = tvdiff(smoothu, 20, 0.03, dx = dx, ε = 1)
    # tvdiff(smoothu, 100, 0.1, dx = dx)
    # 
    # 
    # FDM
    û1 = diff(u) / dx
    # dataset[1] and smoothu are almost equal(rounding errors)
    return û, time, smoothu, û1
end

# need to do this for all datasets
c = [linear(prob.u0, p, t) for t in dataset3[2]] #ideal case
b = calculate_derivatives1(dataset2) #central diffs
# a = calculate_derivatives2(dataset) #tvdiff(smoothu, 100, 0.1, dx = dx)
d = calculate_derivatives2(dataset1) #tvdiff(smoothu, 20, 0.035, dx = dx, ε = 1e-2)
d = calculate_derivatives2(dataset2)
d = calculate_derivatives2(dataset3)
mean(abs2.(c .- b))
mean(abs2.(c .- d[1]))
loss(model, x, y) = mean(abs2.(model(x) .- y));
scatter!(prob.u0 .+ (prob.tspan[2] .- dataset3[2]) .* chainflux1(dataset3[2]')')
loss(chainflux1, dataset3[2]', dataset3[1]')
# mean(abs2.(c[1:24] .- a[4]))
plot(c, label = "ideal deriv")
plot!(b, label = "Centraldiff deriv")
# plot!(a[1], label = "tvdiff(0.1,def) derivatives") 
plot!(d[1], label = "tvdiff(0.035,20) derivatives")
plotly()

# GridTraining , NoiseRobustDiff dataset[2][2]-dataset[2][1] l2std
# 25 points 
ta = range(tspan[1], tspan[2], length = 25)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

time1 = collect(tspan[1]:(1 / 50.0):tspan[2])
physsol = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]
plot(physsol, label = "solution")

# plots from 32(deriv)
# for d
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 2000, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

n2_sol2flux1 = solve(prob, alg)
n2_sol2flux1.estimated_ode_params[1]
# with extra likelihood 
# 10.2011Particles{Float64, 1}

# without extra likelihood 
# 6.25791Particles{Float64, 1}
# 6.29539Particles{Float64, 1}

plot!(n2_sol2flux1.ensemblesol[1], label = "tvdiff(0.035,1) derivpar")
plot(dataset[1])
plot!(physsol1)
# for a
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 2000, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

n2_sol2flux2 = solve(prob, alg)
n2_sol2flux2.estimated_ode_params[1]
# with extra likelihood
# 8.73602Particles{Float64, 1}
# without extra likelihood

plot!(n2_sol2flux2.ensemblesol[1],
    label = "tvdiff(0.1,def) derivatives",
    legend = :outerbottomleft)

# for b
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 2000, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

n2_sol2flux3 = solve(prob, alg)
n2_sol2flux3.estimated_ode_params[1]
plot!(n2_sol2flux3.ensemblesol[1], label = "Centraldiff deriv")

# for c
alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 2000, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

n2_sol2flux4 = solve(prob, alg)
n2_sol2flux4.estimated_ode_params[1]
plot!(n2_sol2flux4.ensemblesol[1], label = "ideal deriv")

# 50 points 

ta = range(tspan[1], tspan[2], length = 50)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

n2_sol2flux11 = solve(prob, alg)
n2_sol2flux11.estimated_ode_params[1]

# 5.90049Particles{Float64, 1}
# 100 points
ta = range(tspan[1], tspan[2], length = 100)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
    draw_samples = 1500, physdt = 1 / 50.0f0,
    priorsNNw = (0.0, 3.0),
    param = [LogNormal(9, 0.5)],
    Metric = DiagEuclideanMetric,
    n_leapfrog = 30, progress = true)

n2_sol2flux111 = solve(prob, alg)
n2_sol2flux111.estimated_ode_params[1]
plot!(n2_sol2flux111.ensemblesol[1])
8.88555Particles{Float64, 1}

# 7.15353Particles{Float64, 1}
# 6.21059 Particles{Float64, 1}
# 6.31836Particles{Float64, 1}
0.1 * p
# ----------------------------------------------------------

# Gives the linear interpolation value at t=3.5

# # Problem 1 with param esimation
# # dataset 0-1 2 percent noise
# p = 6.283185307179586
# # partial_logdensity
# 6.3549Particles{Float64, 1}
# # full log_density
# 6.34667Particles{Float64, 1}

# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2lux.estimated_ode_params[1]

# # dataset 0-1 20 percent noise
# # partial log_density
# 6.30244Particles{Float64, 1}
# # full log_density
# 6.24637Particles{Float64, 1}

# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # dataset 0-2 20percent noise
# # partial log_density
# 6.24948Particles{Float64, 1}
# # full log_density
# 6.26095Particles{Float64, 1}

# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
# linear = (u, p, t) -> cos(p * t)
# tspan = (0.0, 2.0)

# # dataset 0-1 2 percent noise
# p = 6.283185307179586
# # partial_logdensity
# 6.3549Particles{Float64, 1}
# # full log_density
# 6.34667Particles{Float64, 1}

# # dataset 0-1 20 percent noise
# # partial log_density
# 6.30244Particles{Float64, 1}
# # full log_density
# 6.24637Particles{Float64, 1}

# # dataset 0-2 20percent noise
# # partial log_density
# 6.24948Particles{Float64, 1}
# # full log_density
# 6.26095Particles{Float64, 1}

# # dataset 0-2 20percent noise 50 points(above all are 100 points)
# # FuLL log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # partial log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# # i kinda win on 25 points again
# # dataset 0-2 20percent noise 25 points
# # FuLL log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # partial log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # i win with 25 points
# # dataset 0-1 20percent noise 25 points
# # FuLL log_density
# sol2flux.estimated_ode_params[1]
# # new
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # partial log_density
# sol2flux.estimated_ode_params[1]
# # New
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # (9,2.5)(above are (9,0.5))
# # FuLL log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# # just prev was repeat(just change)
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # partial log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # i lose on 0-1,50 points
# # dataset 0-1 20percent noise 50 points
# # FuLL log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]
# # partial log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # (9,2.5) (above are (9,0.5))
# # FuLL log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # partial log_density
# sol2flux.estimated_ode_params[1]
# sol2flux.estimated_ode_params[1]

# # ----------------------------------------------------------
# # Problem 1 with param estimation
# # physdt=1/20, Full likelihood new 0.5*l2std
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n05_sol2flux1 = solve(prob, alg)
# n05_sol2flux1.estimated_ode_params[1] #6.90953 Particles{Float64, 1}
# n05_sol2flux2 = solve(prob, alg)
# n05_sol2flux2.estimated_ode_params[1] #6.82374 Particles{Float64, 1}
# n05_sol2flux3 = solve(prob, alg)
# n05_sol2flux3.estimated_ode_params[1] #6.84465 Particles{Float64, 1}

# using Plots, StatsPlots
# plot(n05_sol2flux3.ensemblesol[1])
# plot!(physsol1)
# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n05_sol2flux11 = solve(prob, alg)
# n05_sol2flux11.estimated_ode_params[1] #7.0262 Particles{Float64, 1}
# n05_sol2flux22 = solve(prob, alg)
# n05_sol2flux22.estimated_ode_params[1] #5.56438 Particles{Float64, 1}
# n05_sol2flux33 = solve(prob, alg)
# n05_sol2flux33.estimated_ode_params[1] #7.27189 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n05_sol2flux111 = solve(prob, alg)
# n05_sol2flux111.estimated_ode_params[1] #6.90549 Particles{Float64, 1}
# n05_sol2flux222 = solve(prob, alg)
# n05_sol2flux222.estimated_ode_params[1] #5.42436 Particles{Float64, 1}
# n05_sol2flux333 = solve(prob, alg)
# n05_sol2flux333.estimated_ode_params[1] #6.05832 Particles{Float64, 1}

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood new 2*l2std
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n2_sol2flux1 = solve(prob, alg)
# n2_sol2flux1.estimated_ode_params[1]#6.9087 Particles{Float64, 1}
# n2_sol2flux2 = solve(prob, alg)
# n2_sol2flux2.estimated_ode_params[1]#6.86507 Particles{Float64, 1}
# n2_sol2flux3 = solve(prob, alg)
# n2_sol2flux3.estimated_ode_params[1]#6.59206 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n2_sol2flux11 = solve(prob, alg)
# n2_sol2flux11.estimated_ode_params[1]#7.3715 Particles{Float64, 1}
# n2_sol2flux22 = solve(prob, alg)
# n2_sol2flux22.estimated_ode_params[1]#9.84477 Particles{Float64, 1}
# n2_sol2flux33 = solve(prob, alg)
# n2_sol2flux33.estimated_ode_params[1]#6.87107 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n2_sol2flux111 = solve(prob, alg)
# n2_sol2flux111.estimated_ode_params[1]#6.60739 Particles{Float64, 1}
# n2_sol2flux222 = solve(prob, alg)
# n2_sol2flux222.estimated_ode_params[1]#7.05923 Particles{Float64, 1}
# n2_sol2flux333 = solve(prob, alg)
# n2_sol2flux333.estimated_ode_params[1]#6.5017 Particles{Float64, 1}

# # ----------------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood new all 2*l2std
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n2all5sol2flux1 = solve(prob, alg)
# n2all5sol2flux1.estimated_ode_params[1]#11.3659 Particles{Float64, 1}
# n2all5sol2flux2 = solve(prob, alg)
# n2all5sol2flux2.estimated_ode_params[1]#6.65634 Particles{Float64, 1}
# n2all5sol2flux3 = solve(prob, alg)
# n2all5sol2flux3.estimated_ode_params[1]#6.61905 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n2all5sol2flux11 = solve(prob, alg)
# n2all5sol2flux11.estimated_ode_params[1]#6.27555 Particles{Float64, 1}
# n2all5sol2flux22 = solve(prob, alg)
# n2all5sol2flux22.estimated_ode_params[1]#6.24352 Particles{Float64, 1}
# n2all5sol2flux33 = solve(prob, alg)
# n2all5sol2flux33.estimated_ode_params[1]#6.33723 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n2all5sol2flux111 = solve(prob, alg)
# n2all5sol2flux111.estimated_ode_params[1] #5.95535 Particles{Float64, 1}
# n2all5sol2flux222 = solve(prob, alg)
# n2all5sol2flux222.estimated_ode_params[1] #5.98301 Particles{Float64, 1}
# n2all5sol2flux333 = solve(prob, alg)
# n2all5sol2flux333.estimated_ode_params[1] #5.9081 Particles{Float64, 1}

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood new all (l2+l22)
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# nall5sol2flux1 = solve(prob, alg)
# nall5sol2flux1.estimated_ode_params[1]#6.54705 Particles{Float64, 1}
# nall5sol2flux2 = solve(prob, alg)
# nall5sol2flux2.estimated_ode_params[1]#6.6967 Particles{Float64, 1}
# nall5sol2flux3 = solve(prob, alg)
# nall5sol2flux3.estimated_ode_params[1]#6.47173 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# nall5sol2flux11 = solve(prob, alg)
# nall5sol2flux11.estimated_ode_params[1]#6.2113 Particles{Float64, 1}
# nall5sol2flux22 = solve(prob, alg)
# nall5sol2flux22.estimated_ode_params[1]#6.10675 Particles{Float64, 1}
# nall5sol2flux33 = solve(prob, alg)
# nall5sol2flux33.estimated_ode_params[1]#6.11541 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# nall5sol2flux111 = solve(prob, alg)
# nall5sol2flux111.estimated_ode_params[1]#6.35224 Particles{Float64, 1}
# nall5sol2flux222 = solve(prob, alg)
# nall5sol2flux222.estimated_ode_params[1]#6.40542 Particles{Float64, 1}
# nall5sol2flux333 = solve(prob, alg)
# nall5sol2flux333.estimated_ode_params[1]#6.44206 Particles{Float64, 1}

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood new 5* (new only l22 mod)
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n5sol2flux1 = solve(prob, alg)
# n5sol2flux1.estimated_ode_params[1]#7.05077 Particles{Float64, 1}
# n5sol2flux2 = solve(prob, alg)
# n5sol2flux2.estimated_ode_params[1]#7.07303 Particles{Float64, 1}
# n5sol2flux3 = solve(prob, alg)
# n5sol2flux3.estimated_ode_params[1]#5.10622 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n5sol2flux11 = solve(prob, alg)
# n5sol2flux11.estimated_ode_params[1]#7.39852 Particles{Float64, 1}
# n5sol2flux22 = solve(prob, alg)
# n5sol2flux22.estimated_ode_params[1]#7.30319 Particles{Float64, 1}
# n5sol2flux33 = solve(prob, alg)
# n5sol2flux33.estimated_ode_params[1]#6.73722 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# n5sol2flux111 = solve(prob, alg)
# n5sol2flux111.estimated_ode_params[1]#7.15996 Particles{Float64, 1}
# n5sol2flux222 = solve(prob, alg)
# n5sol2flux222.estimated_ode_params[1]#7.02949 Particles{Float64, 1}
# n5sol2flux333 = solve(prob, alg)
# n5sol2flux333.estimated_ode_params[1]#6.9393 Particles{Float64, 1}

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood new
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# nsol2flux1 = solve(prob, alg)
# nsol2flux1.estimated_ode_params[1] #5.82707 Particles{Float64, 1}
# nsol2flux2 = solve(prob, alg)
# nsol2flux2.estimated_ode_params[1] #4.81534 Particles{Float64, 1}
# nsol2flux3 = solve(prob, alg)
# nsol2flux3.estimated_ode_params[1] #5.52965 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# nsol2flux11 = solve(prob, alg)
# nsol2flux11.estimated_ode_params[1] #7.04027 Particles{Float64, 1}
# nsol2flux22 = solve(prob, alg)
# nsol2flux22.estimated_ode_params[1] #7.17588 Particles{Float64, 1}
# nsol2flux33 = solve(prob, alg)
# nsol2flux33.estimated_ode_params[1] #6.94495 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# nsol2flux111 = solve(prob, alg)
# nsol2flux111.estimated_ode_params[1] #6.06608 Particles{Float64, 1}
# nsol2flux222 = solve(prob, alg)
# nsol2flux222.estimated_ode_params[1] #6.84726 Particles{Float64, 1}
# nsol2flux333 = solve(prob, alg)
# nsol2flux333.estimated_ode_params[1] #6.83463 Particles{Float64, 1}

# # ----------------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux1 = solve(prob, alg)
# sol2flux1.estimated_ode_params[1] #6.71397 Particles{Float64, 1} 6.37604 Particles{Float64, 1}
# sol2flux2 = solve(prob, alg)
# sol2flux2.estimated_ode_params[1] #6.73509 Particles{Float64, 1} 6.21692 Particles{Float64, 1}
# sol2flux3 = solve(prob, alg)
# sol2flux3.estimated_ode_params[1] #6.65453 Particles{Float64, 1} 6.23153 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux11 = solve(prob, alg)
# sol2flux11.estimated_ode_params[1] #6.23443 Particles{Float64, 1} 6.30635 Particles{Float64, 1}
# sol2flux22 = solve(prob, alg)
# sol2flux22.estimated_ode_params[1] #6.18879 Particles{Float64, 1} 6.30099 Particles{Float64, 1}
# sol2flux33 = solve(prob, alg)
# sol2flux33.estimated_ode_params[1] #6.22773 Particles{Float64, 1} 6.30671 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux111 = solve(prob, alg)
# sol2flux111.estimated_ode_params[1] #6.15832 Particles{Float64, 1} 6.35453 Particles{Float64, 1}
# sol2flux222 = solve(prob, alg)
# sol2flux222.estimated_ode_params[1] #6.16968 Particles{Float64, 1}6.31125 Particles{Float64, 1}
# sol2flux333 = solve(prob, alg)
# sol2flux333.estimated_ode_params[1] #6.12466 Particles{Float64, 1} 6.26514 Particles{Float64, 1}

# # ----------------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, partial likelihood
# # 25 points
# ta = range(tspan[1], tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux1_p = solve(prob, alg)
# sol2flux1_p.estimated_ode_params[1] #5.74065 Particles{Float64, 1} #6.83683 Particles{Float64, 1}
# sol2flux2_p = solve(prob, alg)
# sol2flux2_p.estimated_ode_params[1] #9.82504 Particles{Float64, 1} #6.14568 Particles{Float64, 1}
# sol2flux3_p = solve(prob, alg)
# sol2flux3_p.estimated_ode_params[1] #5.75075 Particles{Float64, 1} #6.08579 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux11_p = solve(prob, alg)
# sol2flux11_p.estimated_ode_params[1] #6.19414 Particles{Float64, 1} #6.04621 Particles{Float64, 1}
# sol2flux22_p = solve(prob, alg)
# sol2flux22_p.estimated_ode_params[1] #6.15227 Particles{Float64, 1} #6.29086 Particles{Float64, 1}
# sol2flux33_p = solve(prob, alg)
# sol2flux33_p.estimated_ode_params[1] #6.19048 Particles{Float64, 1} #6.12516 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol2flux111_p = solve(prob, alg)
# sol2flux111_p.estimated_ode_params[1] #6.51608 Particles{Float64, 1}# 6.42945Particles{Float64, 1}
# sol2flux222_p = solve(prob, alg)
# sol2flux222_p.estimated_ode_params[1] #6.4875 Particles{Float64, 1} # 6.44524Particles{Float64, 1}
# sol2flux333_p = solve(prob, alg)
# sol2flux333_p.estimated_ode_params[1] #6.51679 Particles{Float64, 1}# 6.43152Particles{Float64, 1}

# # ---------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood, dataset(1.0-2.0)
# # 25 points
# ta = range(1.0, tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1flux1 = solve(prob, alg)
# sol1flux1.estimated_ode_params[1] #6.35164 Particles{Float64, 1}
# sol1flux2 = solve(prob, alg)
# sol1flux2.estimated_ode_params[1] #6.30919 Particles{Float64, 1}
# sol1flux3 = solve(prob, alg)
# sol1flux3.estimated_ode_params[1] #6.33554 Particles{Float64, 1}

# # 50 points
# ta = range(1.0, tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1flux11 = solve(prob, alg)
# sol1flux11.estimated_ode_params[1] #6.39769 Particles{Float64, 1}
# sol1flux22 = solve(prob, alg)
# sol1flux22.estimated_ode_params[1] #6.43924 Particles{Float64, 1}
# sol1flux33 = solve(prob, alg)
# sol1flux33.estimated_ode_params[1] #6.4697 Particles{Float64, 1}

# # 100 points
# ta = range(1.0, tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1flux111 = solve(prob, alg)
# sol1flux111.estimated_ode_params[1] #6.27812 Particles{Float64, 1}
# sol1flux222 = solve(prob, alg)
# sol1flux222.estimated_ode_params[1] #6.19278 Particles{Float64, 1}
# sol1flux333 = solve(prob, alg)
# sol1flux333.estimated_ode_params[1] # 9.68244Particles{Float64, 1} (first try) # 6.23969 Particles{Float64, 1}(second try)

# # ----------------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, partial likelihood, dataset(1.0-2.0)
# # 25 points
# ta = range(1.0, tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1flux1_p = solve(prob, alg)
# sol1flux1_p.estimated_ode_params[1]#6.36269 Particles{Float64, 1}

# sol1flux2_p = solve(prob, alg)
# sol1flux2_p.estimated_ode_params[1]#6.34685 Particles{Float64, 1}

# sol1flux3_p = solve(prob, alg)
# sol1flux3_p.estimated_ode_params[1]#6.31421 Particles{Float64, 1}

# # 50 points
# ta = range(1.0, tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1flux11_p = solve(prob, alg)
# sol1flux11_p.estimated_ode_params[1] #6.15725 Particles{Float64, 1}

# sol1flux22_p = solve(prob, alg)
# sol1flux22_p.estimated_ode_params[1] #6.18145 Particles{Float64, 1}

# sol1flux33_p = solve(prob, alg)
# sol1flux33_p.estimated_ode_params[1] #6.21905 Particles{Float64, 1}

# # 100 points
# ta = range(1.0, tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1flux111_p = solve(prob, alg)
# sol1flux111_p.estimated_ode_params[1]#6.13481 Particles{Float64, 1}

# sol1flux222_p = solve(prob, alg)
# sol1flux222_p.estimated_ode_params[1]#9.68555 Particles{Float64, 1}

# sol1flux333_p = solve(prob, alg)
# sol1flux333_p.estimated_ode_params[1]#6.1477 Particles{Float64, 1}

# # -----------------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, partial likelihood, dataset(1-2), again but different density
# # 12 points
# ta = range(1.0, tspan[2], length = 12)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol3flux1_p = solve(prob, alg)
# sol3flux1_p.estimated_ode_params[1]#6.50048 Particles{Float64, 1}
# sol3flux2_p = solve(prob, alg)
# sol3flux2_p.estimated_ode_params[1]#6.57597 Particles{Float64, 1}
# sol3flux3_p = solve(prob, alg)
# sol3flux3_p.estimated_ode_params[1]#6.24487 Particles{Float64, 1}

# # 25 points
# ta = range(1.0, tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol3flux11_p = solve(prob, alg)
# sol3flux11_p.estimated_ode_params[1]#6.53093 Particles{Float64, 1}

# sol3flux22_p = solve(prob, alg)
# sol3flux22_p.estimated_ode_params[1]#6.32744 Particles{Float64, 1}

# sol3flux33_p = solve(prob, alg)
# sol3flux33_p.estimated_ode_params[1]#6.49175 Particles{Float64, 1}

# # 50 points
# ta = range(1.0, tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol3flux111_p = solve(prob, alg)
# sol3flux111_p.estimated_ode_params[1]#6.4455 Particles{Float64, 1}
# sol3flux222_p = solve(prob, alg)
# sol3flux222_p.estimated_ode_params[1]#6.40736 Particles{Float64, 1}
# sol3flux333_p = solve(prob, alg)
# sol3flux333_p.estimated_ode_params[1]#6.46214 Particles{Float64, 1}

# # ---------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, partial likelihood, dataset(0-1)
# # 25 points
# ta = range(tspan[1], 1.0, length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol0flux1_p = solve(prob, alg)
# sol0flux1_p.estimated_ode_params[1]#7.12625 Particles{Float64, 1}
# sol0flux2_p = solve(prob, alg)
# sol0flux2_p.estimated_ode_params[1]#8.40948 Particles{Float64, 1}
# sol0flux3_p = solve(prob, alg)
# sol0flux3_p.estimated_ode_params[1]#7.18768 Particles{Float64, 1}

# # 50 points
# ta = range(tspan[1], 1.0, length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol0flux11_p = solve(prob, alg)
# sol0flux11_p.estimated_ode_params[1]#6.23707 Particles{Float64, 1}
# sol0flux22_p = solve(prob, alg)
# sol0flux22_p.estimated_ode_params[1]#6.09728 Particles{Float64, 1}
# sol0flux33_p = solve(prob, alg)
# sol0flux33_p.estimated_ode_params[1]#6.12971 Particles{Float64, 1}

# # 100 points
# ta = range(tspan[1], 1.0, length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [LogNormal(9, 0.5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol0flux111_p = solve(prob, alg)
# sol0flux111_p.estimated_ode_params[1]#5.99039 Particles{Float64, 1}
# sol0flux222_p = solve(prob, alg)
# sol0flux222_p.estimated_ode_params[1]#5.89609 Particles{Float64, 1}
# sol0flux333_p = solve(prob, alg)
# sol0flux333_p.estimated_ode_params[1]#5.91923 Particles{Float64, 1}

# # ---------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, Full likelihood, dataset(1.0-2.0), Normal(12,5) distri prior
# # 25 points
# ta = range(1.0, tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 6.0),
#     param = [Normal(12, 5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1f1 = solve(prob, alg)
# sol1f1.estimated_ode_params[1]
# # 10.9818Particles{Float64, 1}
# sol1f2 = solve(prob, alg)
# sol1f2.estimated_ode_params[1]
# # sol1f3 = solve(prob, alg)
# # sol1f3.estimated_ode_params[1]

# # 50 points
# ta = range(1.0, tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 6.0),
#     param = [Normal(12, 5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1f11 = solve(prob, alg)
# sol1f11.estimated_ode_params[1]
# sol1f22 = solve(prob, alg)
# sol1f22.estimated_ode_params[1]
# # sol1f33 = solve(prob, alg)
# # sol1f33.estimated_ode_params[1]

# # 100 points
# ta = range(1.0, tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 6.0),
#     param = [Normal(12, 5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1f111 = solve(prob, alg)
# sol1f111.estimated_ode_params[1]
# sol1f222 = solve(prob, alg)
# sol1f222.estimated_ode_params[1]
# # sol1f333 = solve(prob, alg)
# # sol1f333.estimated_ode_params[1]

# # ----------------------------------------------------------

# # ----------------------------------------------------------
# # physdt=1/20, partial likelihood, dataset(1.0-2.0), Normal(12,5) distri prior
# # 25 points
# ta = range(1.0, tspan[2], length = 25)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [Normal(12, 5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1f1_p = solve(prob, alg)
# sol1f1_p.estimated_ode_params[1]
# sol1f2_p = solve(prob, alg)
# sol1f2_p.estimated_ode_params[1]
# sol1f3_p = solve(prob, alg)
# sol1f3_p.estimated_ode_params[1]

# # 50 points
# ta = range(1.0, tspan[2], length = 50)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [Normal(12, 5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1f11_p = solve(prob, alg)
# sol1f11_p.estimated_ode_params[1]
# sol1f22_p = solve(prob, alg)
# sol1f22_p.estimated_ode_params[1]
# sol1f33_p = solve(prob, alg)
# sol1f33_p.estimated_ode_params[1]

# # 100 points
# ta = range(1.0, tspan[2], length = 100)
# u = [linear_analytic(u0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂, time]
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

# alg = NeuralPDE.BNNODE(chainflux1, dataset = dataset,
#     draw_samples = 1500, physdt = 1 / 50.0f0,
#     priorsNNw = (0.0, 3.0),
#     param = [Normal(12, 5)],
#     Metric = DiagEuclideanMetric,
#     n_leapfrog = 30, progress = true)

# sol1f111_p = solve(prob, alg)
# sol1f111_p.estimated_ode_params[1]
# sol1f222_p = solve(prob, alg)
# sol1f222_p.estimated_ode_params[1]
# sol1f333_p = solve(prob, alg)
# sol1f333_p.estimated_ode_params[1]

# # ----------------------------------------------------------

# plot!(title = "9,2.5 50 training 2>full,1>partial")

# p
# param1
# # (lux chain)
# @prob mean(abs.(physsol2 .- sol3lux_pestim.ensemblesol[1])) < 8e-2

# # estimated parameters(lux chain)
# param1 = sol3lux_pestim.estimated_ode_params[1]
# @test abs(param1 - p) < abs(0.35 * p)

# p
# param1

# # # my suggested Loss likelihood part
# # #  + L2loss2(Tar, θ)
# # # My suggested extra loss function
# # function L2loss2(Tar::LogTargetDensity, θ)
# #     f = Tar.prob.f

# #     # parameter estimation chosen or not
# #     if Tar.extraparams > 0
# #         dataset = Tar.dataset

# #         # Timepoints to enforce Physics
# #         dataset = Array(reduce(hcat, dataset)')
# #         t = dataset[end, :]
# #         û = dataset[1:(end - 1), :]

# #         ode_params = Tar.extraparams == 1 ?
# #                      θ[((length(θ) - Tar.extraparams) + 1):length(θ)][1] :
# #                      θ[((length(θ) - Tar.extraparams) + 1):length(θ)]

# #         if length(û[:, 1]) == 1
# #             physsol = [f(û[:, i][1],
# #                 ode_params,
# #                 t[i])
# #                        for i in 1:length(û[1, :])]
# #         else
# #             physsol = [f(û[:, i],
# #                 ode_params,
# #                 t[i])
# #                        for i in 1:length(û[1, :])]
# #         end
# #         #form of NN output matrix output dim x n
# #         deri_physsol = reduce(hcat, physsol)

# #         #   OG deriv(basically gradient matching in case of an ODEFunction)
# #         # in case of PDE or general ODE we would want to reduce residue of f(du,u,p,t)
# #         # if length(û[:, 1]) == 1
# #         #     deri_sol = [f(û[:, i][1],
# #         #         Tar.prob.p,
# #         #         t[i])
# #         #                 for i in 1:length(û[1, :])]
# #         # else
# #         #     deri_sol = [f(û[:, i],
# #         #         Tar.prob.p,
# #         #         t[i])
# #         #                 for i in 1:length(û[1, :])]
# #         # end
# #         # deri_sol = reduce(hcat, deri_sol)
# #         derivatives = calculate_derivatives(Tar.dataset)
# #         deri_sol = reduce(hcat, derivatives)

# #         physlogprob = 0
# #         for i in 1:length(Tar.prob.u0)
# #             # can add phystd[i] for u[i]
# #             physlogprob += logpdf(MvNormal(deri_physsol[i, :],
# #                     LinearAlgebra.Diagonal(map(abs2,
# #                         Tar.l2std[i] .*
# #                         ones(length(deri_sol[i, :]))))),
# #                 deri_sol[i, :])
# #         end
# #         return physlogprob
# #     else
# #         return 0
# #     end
# # end

# # function calculate_derivatives(dataset)
# #     x̂, time = dataset
# #     num_points = length(x̂)

# #     # Initialize an array to store the derivative values.
# #     derivatives = similar(x̂)

# #     for i in 2:(num_points - 1)
# #         # Calculate the first-order derivative using central differences.
# #         Δt_forward = time[i + 1] - time[i]
# #         Δt_backward = time[i] - time[i - 1]

# #         derivative = (x̂[i + 1] - x̂[i - 1]) / (Δt_forward + Δt_backward)

# #         derivatives[i] = derivative
# #     end

# #     # Derivatives at the endpoints can be calculated using forward or backward differences.
# #     derivatives[1] = (x̂[2] - x̂[1]) / (time[2] - time[1])
# #     derivatives[end] = (x̂[end] - x̂[end - 1]) / (time[end] - time[end - 1])

# #     return derivatives
# # end

# size(dataset[1])
# # Problem 1 with param estimation(flux,lux)
# # Normal
# # 6.20311 Particles{Float64, 1},6.21746Particles{Float64, 1}
# # better
# # 6.29093Particles{Float64, 1}, 6.27925Particles{Float64, 1}
# # Non ideal case
# # 6.14861Particles{Float64, 1}, 
# sol2flux.estimated_ode_params
# sol2lux.estimated_ode_params[1]
# p
# size(sol3flux_pestim.ensemblesol[2])
# plott = sol3flux_pestim.ensemblesol[1]
# using StatsPlots
# plotly()
# plot(t, sol3flux_pestim.ensemblesol[1])

# function calculate_derivatives(dataset)
#     x̂, time = dataset
#     num_points = length(x̂)

#     # Initialize an array to store the derivative values.
#     derivatives = similar(x̂)

#     for i in 2:(num_points - 1)
#         # Calculate the first-order derivative using central differences.
#         Δt_forward = time[i + 1] - time[i]
#         Δt_backward = time[i] - time[i - 1]

#         derivative = (x̂[i + 1] - x̂[i - 1]) / (Δt_forward + Δt_backward)

#         derivatives[i] = derivative
#     end

#     # Derivatives at the endpoints can be calculated using forward or backward differences.
#     derivatives[1] = (x̂[2] - x̂[1]) / (time[2] - time[1])
#     derivatives[end] = (x̂[end] - x̂[end - 1]) / (time[end] - time[end - 1])

#     return derivatives
# end

# # Example usage:
# # dataset = [x̂, time]
# derivatives = calculate_derivatives(dataset)
# dataset[1]
# # Access derivative values at specific time points as needed.

# # # 9,0.5
# # 0.09894916260292887
# # 0.09870335436072103
# # 0.08398556878067913
# # 0.10109070099105527
# # 0.09122683737517055
# # 0.08614958011892977
# # mean(abs.(x̂ .- meanscurve1)) #0.017112298305523976
# # mean(abs.(physsol1 .- meanscurve1)) #0.004038636894341354
# # # 9,4(little worse)
# # mean(abs.(x̂ .- meanscurve1))#0.01800876370000113
# # mean(abs.(physsol1 .- meanscurve1))#0.007285681280600875
# # # 30,30
# # mean(abs.(x̂ .- meanscurve1)) #0.10599926120358046
# # mean(abs.(physsol1 .- meanscurve1)) #0.10375554193397989
# # # 30,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.10160824458252521
# # mean(abs.(physsol1 .- meanscurve1)) #0.09999942538357891

# # # ------------------------------------------------normale
# # # 9,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.0333356493928835
# # mean(abs.(physsol1 .- meanscurve1)) #0.02721733876400459
# # # 9,4(little worse)
# # mean(abs.(x̂ .- meanscurve1)) #0.020734206709433347
# # mean(abs.(physsol1 .- meanscurve1)) #0.012502850740700212
# # # 30,30
# # mean(abs.(x̂ .- meanscurve1)) #0.10615859683094729
# # mean(abs.(physsol1 .- meanscurve1)) #0.10508141153722575
# # # 30,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.10833514946031565
# # mean(abs.(physsol1 .- meanscurve1)) #0.10668470203219232

# # # 9,0.5
# # 10.158108285475553
# # 10.207234384538026
# # 10.215000657664852
# # 10.213817644016174
# # 13.380030074088719
# # 13.348906350967326

# # 6.952731422892041

# # # All losses
# # 10.161478523326277
# # # L2 losses 1
# # 9.33312996960278
# # # L2 losses 2
# # 10.217417241370631

# # mean([fhsamples1[i][26] for i in 500:1000]) #6.245045767509431
# # p #6.283185307179586
# # # 9,4
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.212522300650451
# # # 30,30
# # mean([fhsamples1[i][23] for i in 500:1000]) #35.328636809737695
# # # 30,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #35.232963812125654

# # # ---------------------------------------normale
# # # 9,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.547771572198114
# # p #6.283185307179586
# # # 9,4
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.158906185002702
# # # 30,30
# # mean([fhsamples1[i][23] for i in 500:1000]) #29.210400972620185
# # # 30,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #29.153845019454522

# # # ----------------more dataset normale -----------------------------
# # # 9,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.271141178216537
# # p #6.283185307179586
# # # 9,4
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.241144692919369
# # # 30,30
# # mean([fhsamples1[i][23] for i in 500:1000]) #29.124480447973127
# # # 30,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #29.07838011629903

# # # 9,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.016551602015599295
# # mean(abs.(physsol1 .- meanscurve1)) #0.0021488618484224245
# # # 9,4(little worse)
# # mean(abs.(x̂ .- meanscurve1)) #0.017022725082640747
# # mean(abs.(physsol1 .- meanscurve1)) #0.004339761917100232
# # # 30,30
# # mean(abs.(x̂ .- meanscurve1)) #0.09668785317864312
# # mean(abs.(physsol1 .- meanscurve1)) #0.09430712337543362
# # # 30,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.09958118358974392
# # mean(abs.(physsol1 .- meanscurve1)) #0.09717454226368502

# # # ----------------more dataset special -----------------------------
# # # 9,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.284355334485365
# # p #6.283185307179586
# # # 9,4
# # mean([fhsamples1[i][23] for i in 500:1000]) #6.259238106698602
# # # 30,30
# # mean([fhsamples1[i][23] for i in 500:1000]) #29.139808934336987
# # # 30,0.5
# # mean([fhsamples1[i][23] for i in 500:1000]) #29.03921327641226

# # # 9,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.016627231605546876
# # mean(abs.(physsol1 .- meanscurve1)) #0.0020311429130039564
# # # 9,4(little worse)
# # mean(abs.(x̂ .- meanscurve1)) #0.016650324577507352
# # mean(abs.(physsol1 .- meanscurve1)) #0.0027537543411154677
# # # 30,30
# # mean(abs.(x̂ .- meanscurve1)) #0.09713187937270151
# # mean(abs.(physsol1 .- meanscurve1)) #0.09317278450371556
# # # 30,0.5
# # mean(abs.(x̂ .- meanscurve1)) #0.09550234866855814
# # mean(abs.(physsol1 .- meanscurve1)) #0.09317278450371556

# # using Plots, StatsPlots
# # plotly()

# # ---------------------------------------------------------
# # # # Distribution abstract in wrapper, dataset Float64
# # # 268.651 s (206393690 allocations: 388.71 GiB)
# # # 318.170551 seconds (206.29 M allocations: 388.453 GiB, 20.83% gc time)

# # # # Above with dataset Real subtype
# # # 326.201 s (206327409 allocations: 388.42 GiB)
# # # 363.189370 seconds (206.25 M allocations: 387.975 GiB, 15.77% gc time)
# # # 306.171 s (206321277 allocations: 388.55 GiB)
# # # 356.180699 seconds (206.43 M allocations: 388.361 GiB, 13.77% gc time)

# # # # Above with dataset AbstractFloat subtype
# # # 290.751187 seconds (205.94 M allocations: 387.955 GiB, 12.92% gc time)
# # # 296.319815 seconds (206.38 M allocations: 388.730 GiB, 12.69% gc time)

# # # # ODEProblem float64 dtaset and vector distri inside
# # #   273.169 s (206128318 allocations: 388.40 GiB)
# # #   274.059531 seconds (205.91 M allocations: 387.953 GiB, 12.77% gc time)

# # # #   Dataset float64 inside and vector distri outsude
# # #   333.603 s (206251143 allocations: 388.41 GiB)
# # # 373.377222 seconds (206.11 M allocations: 387.968 GiB, 13.25% gc time)
# # #   359.745 s (206348301 allocations: 388.41 GiB)
# # # 357.813114 seconds (206.31 M allocations: 388.354 GiB, 13.54% gc time)

# # # # Dataset float64 inside and vector distri inside
# # #   326.437 s (206253571 allocations: 388.41 GiB)
# # #   290.334083 seconds (205.92 M allocations: 387.954 GiB, 13.82% gc time)

# # # # current setting
# # # 451.304 s (206476927 allocations: 388.43 GiB)
# # # 384.532732 seconds (206.22 M allocations: 387.976 GiB, 13.17% gc time)
# # # 310.223 s (206332558 allocations: 388.63 GiB)
# # # 344.243889 seconds (206.34 M allocations: 388.409 GiB, 13.84% gc time)
# # # 357.457737 seconds (206.66 M allocations: 389.064 GiB, 18.16% gc time)

# # # # shit setup
# # #   325.595 s (206283732 allocations: 388.41 GiB)
# # # 334.248753 seconds (206.06 M allocations: 387.964 GiB, 12.60% gc time)
# # #   326.011 s (206370857 allocations: 388.56 GiB)
# # # 327.203339 seconds (206.29 M allocations: 388.405 GiB, 12.92% gc time)

# # # # in wrapper Distribution prior, insiade FLOAT64 DATASET
# # # 325.158167 seconds (205.97 M allocations: 387.958 GiB, 15.07% gc time) 
# # #   429.536 s (206476324 allocations: 388.43 GiB)
# # #   527.364 s (206740343 allocations: 388.58 GiB)

# # # #   wrapper Distribtuion, inside Float64
# # # 326.017 s (206037971 allocations: 387.96 GiB)
# # # 347.424730 seconds (206.45 M allocations: 388.532 GiB, 12.92% gc time)

# # # 439.047568 seconds (284.24 M allocations: 392.598 GiB, 15.25% gc time, 14.36% compilation time: 0% of which was recompilation)
# # # 375.472142 seconds (206.40 M allocations: 388.529 GiB, 14.93% gc time)
# # # 374.888820 seconds (206.34 M allocations: 388.346 GiB, 14.09% gc time)
# # # 363.719611 seconds (206.39 M allocations: 388.581 GiB, 15.08% gc time)
# # # # inside Distribtion, instide Float64
# # #   310.238 s (206324249 allocations: 388.53 GiB)
# # #   308.991494 seconds (206.34 M allocations: 388.549 GiB, 14.01% gc time)
# # #   337.442 s (206280712 allocations: 388.36 GiB)
# # #   299.983096 seconds (206.29 M allocations: 388.512 GiB, 17.14% gc time)

# # #   394.924357 seconds (206.27 M allocations: 388.337 GiB, 23.68% gc time)
# # # 438.204179 seconds (206.39 M allocations: 388.470 GiB, 23.84% gc time)
# # #   376.626914 seconds (206.46 M allocations: 388.693 GiB, 18.72% gc time)
# # # 286.863795 seconds (206.14 M allocations: 388.370 GiB, 18.80% gc time)
# # #   285.556929 seconds (206.22 M allocations: 388.371 GiB, 17.04% gc time)
# # #   291.471662 seconds (205.96 M allocations: 388.068 GiB, 19.85% gc time)

# # # 495.814341 seconds (284.62 M allocations: 392.622 GiB, 12.56% gc time, 10.96% compilation time: 0% of which was recompilation)
# # # 361.530617 seconds (206.36 M allocations: 388.526 GiB, 14.98% gc time)
# # # 348.576065 seconds (206.22 M allocations: 388.337 GiB, 15.01% gc time)
# # # 374.575609 seconds (206.45 M allocations: 388.586 GiB, 14.65% gc time)
# # # 314.223008 seconds (206.23 M allocations: 388.411 GiB, 14.63% gc time)

# # PROBLEM-3 LOTKA VOLTERRA EXAMPLE [WIP] (WITH PARAMETER ESTIMATION)(will be put in tutorial page)
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

# u0 = [1.0, 1.0]
# p = [1.5, 1.0, 3.0, 1.0]
# tspan = (0.0, 6.0)
# prob = ODEProblem(lotka_volterra, u0, tspan, p)
# solution = solve(prob, Tsit5(); saveat = 0.05)

# as = reduce(hcat, solution.u)
# as[1, :]
# # Plot simulation.
# time = solution.t
# u = hcat(solution.u...)
# # BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
# x = u[1, :] + 0.5 * randn(length(u[1, :]))
# y = u[2, :] + 0.5 * randn(length(u[1, :]))
# dataset = [x[1:50], y[1:50], time[1:50]]
# # scatter!(time, [x, y])
# # scatter!(dataset[3], [dataset[2], dataset[1]])

# # NN has 2 outputs as u -> [dx,dy]
# chainlux1 = Lux.Chain(Lux.Dense(1, 6, Lux.tanh), Lux.Dense(6, 6, Lux.tanh),
#     Lux.Dense(6, 2))
# chainflux1 = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh), Flux.Dense(6, 2))

# # fh_mcmc_chainflux1, fhsamplesflux1, fhstatsflux1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
# #                                                                           dataset = dataset,
# #                                                                           draw_samples = 1000,
# #                                                                           l2std = [
# #                                                                               0.05,
# #                                                                               0.05,
# #                                                                           ],
# #                                                                           phystd = [
# #                                                                               0.05,
# #                                                                               0.05,
# #                                                                           ],
# #                                                                           priorsNNw = (0.0,
# #          

# #   3.0))

# # check if NN output is more than 1
# # numoutput = size(luxar[1])[1]
# # if numoutput > 1
# #     # Initialize a vector to store the separated outputs for each output dimension
# #     output_matrices = [Vector{Vector{Float32}}() for _ in 1:numoutput]

# #     # Loop through each element in the `as` vector
# #     for element in as
# #         for i in 1:numoutput
# #             push!(output_matrices[i], element[i, :])  # Append the i-th output (i-th row) to the i-th output_matrices
# #         end
# #     end

# #     ensemblecurves = Vector{}[]
# #     for r in 1:numoutput
# #         br = hcat(output_matrices[r]...)'
# #         ensemblecurve = prob.u0[r] .+
# #                         [Particles(br[:, i]) for i in 1:length(t)] .*
# #                         (t .- prob.tspan[1])
# #         push!(ensemblecurves, ensemblecurve)
# #     end

# # else
# #     # ensemblecurve = prob.u0 .+
# #     #                 [Particles(reduce(vcat, luxar)[:, i]) for i in 1:length(t)] .*
# #     #                 (t .- prob.tspan[1])
# #     print("yuh")
# # end

# # fhsamplesflux2
# # nnparams = length(init1)
# # estimnnparams = [Particles(reduce(hcat, fhsamplesflux2)[i, :]) for i in 1:nnparams]
# # ninv=4
# # estimated_params = [Particles(reduce(hcat, fhsamplesflux2[(end - ninv + 1):end])[i, :])
# #                     for i in (nnparams + 1):(nnparams + ninv)]
# # output_matrices[r]
# # br = hcat(output_matrices[r]...)'

# # br[:, 1]

# # [Particles(br[:, i]) for i in 1:length(t)]
# # prob.u0
# # [Particles(br[:, i]) for i in 1:length(t)] .*
# # (t .- prob.tspan[1])

# # ensemblecurve = prob.u0[r] .+
# #                 [Particles(br[:, i]) for i in 1:length(t)] .*
# #                 (t .- prob.tspan[1])
# # push!(ensemblecurves, ensemblecurve)

# using StatsPlots
# plotly()
# plot(t, ensemblecurve)
# plot(t, ensemblecurves[1])
# plot!(t, ensemblecurves[2])
# ensemblecurve
# ensemblecurves[1]
# fh_mcmc_chainflux2, fhsamplesflux2, fhstatsflux2 = ahmc_bayesian_pinn_ode(prob, chainflux1,
#     dataset = dataset,
#     draw_samples = 1000,
#     l2std = [
#         0.05,
#         0.05,
#     ],
#     phystd = [
#         0.05,
#         0.05,
#     ],
#     priorsNNw = (0.0,
#         3.0),
#     param = [
#         Normal(1.5,
#             0.5),
#         Normal(1.2,
#             0.5),
#         Normal(3.3,
#             0.5),
#         Normal(1.4,
#             0.5),
#     ], progress = true)

# alg = NeuralPDE.BNNODE(chainflux1,
#     dataset = dataset,
#     draw_samples = 1000,
#     l2std = [
#         0.05,
#         0.05,
#     ],
#     phystd = [
#         0.05,
#         0.05,
#     ],
#     priorsNNw = (0.0,
#         3.0),
#     param = [
#         Normal(4.5,
#             5),
#         Normal(7,
#             2),
#         Normal(5,
#             2),
#         Normal(-4,
#             6),
#     ],
#     n_leapfrog = 30, progress = true)

# sol3flux_pestim = solve(prob, alg)

# # OG PARAM VALUES
# [1.5, 1.0, 3.0, 1.0]
# # less
# # [1.34, 7.51, 2.54, -2.55]
# # better
# # [1.48, 0.993, 2.77, 0.954]

# sol3flux_pestim.es
# sol3flux_pestim.estimated_ode_params
# # fh_mcmc_chainlux1, fhsampleslux1, fhstatslux1 = ahmc_bayesian_pinn_ode(prob, chainlux1,
# #                                                                        dataset = dataset,
# #                                                                        draw_samples = 1000,
# #                                                                        l2std = [0.05, 0.05],
# #                                                                        phystd = [
# #                                                                            0.05,
# #                                                                            0.05,
# #                                                                        ],
# #                                                                        priorsNNw = (0.0,
# #                                                                                     3.0))

# # fh_mcmc_chainlux2, fhsampleslux2, fhstatslux2 = ahmc_bayesian_pinn_ode(prob, chainlux1,
# #                                                                        dataset = dataset,
# #                                                                        draw_samples = 1000,
# #                                                                        l2std = [0.05, 0.05],
# #                                                                        phystd = [
# #                                                                            0.05,
# #                                                                            0.05,
# #                                                                        ],
# #                                                                        priorsNNw = (0.0,
# #                                                                                     3.0),
# #                                                                        param = [
# #                                                                            Normal(1.5, 0.5),
# #                                                                            Normal(1.2, 0.5),
# #                                                                            Normal(3.3, 0.5),
# #                                                                            Normal(1.4, 0.5),
# #                                                                        ])

# init1, re1 = destructure(chainflux1)
# θinit, st = Lux.setup(Random.default_rng(), chainlux1)
# #   PLOT testing points
# t = time
# p = prob.p
# collect(Float64, vcat(ComponentArrays.ComponentArray(θinit)))
# collect(Float64, ComponentArrays.ComponentArray(θinit))
# # Mean of last 1000 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
# out = re1.([fhsamplesflux1[i][1:68] for i in 500:1000])
# yu = [out[i](t') for i in eachindex(out)]

# function getensemble(yu, num_models)
#     num_rows, num_cols = size(yu[1])
#     row_means = zeros(Float32, num_rows, num_cols)
#     for i in 1:num_models
#         row_means .+= yu[i]
#     end
#     row_means ./ num_models
# end
# fluxmean = getensemble(yu, length(out))
# meanscurve1_1 = prob.u0 .+ (t' .- prob.tspan[1]) .* fluxmean
# mean(abs.(u .- meanscurve1_1))

# plot!(t, physsol1)
# @test mean(abs2.(x̂ .- meanscurve1_1)) < 2e-2
# @test mean(abs2.(physsol1 .- meanscurve1)) < 2e-2
# @test mean(abs2.(x̂ .- meanscurve2)) < 3e-3
# @test mean(abs2.(physsol1 .- meanscurve2)) < 2e-3

# out = re1.([fhsamplesflux2[i][1:68] for i in 500:1000])
# yu = collect(out[i](t') for i in eachindex(out))
# fluxmean = getensemble(yu, length(out))
# meanscurve1_2 = prob.u0 .+ (t' .- prob.tspan[1]) .* fluxmean
# mean(abs.(u .- meanscurve1_2))

# @test mean(abs2.(x̂ .- meanscurve1)) < 2e-2
# @test mean(abs2.(physsol1 .- meanscurve1)) < 2e-2
# @test mean(abs2.(x̂ .- meanscurve2)) < 3e-3
# @test mean(abs2.(physsol1 .- meanscurve2)) < 2e-3

# θ = [vector_to_parameters(fhsampleslux1[i][1:(end - 4)], θinit) for i in 500:1000]
# luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
# luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
# meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

# @test mean(abs2.(x̂ .- meanscurve1)) < 2e-2
# @test mean(abs2.(physsol1 .- meanscurve1)) < 2e-2
# @test mean(abs2.(x̂ .- meanscurve2)) < 3e-3
# @test mean(abs2.(physsol1 .- meanscurve2)) < 2e-3

# θ = [vector_to_parameters(fhsampleslux2[i][1:(end - 4)], θinit) for i in 500:1000]
# luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
# luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
# meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

# @test mean(abs2.(x̂ .- meanscurve1)) < 2e-2
# @test mean(abs2.(physsol1 .- meanscurve1)) < 2e-2
# @test mean(abs2.(x̂ .- meanscurve2)) < 3e-3
# @test mean(abs2.(physsol1 .- meanscurve2)) < 2e-3

# # # ESTIMATED ODE PARAMETERS (NN1 AND NN2)
# @test abs(p - mean([fhsamplesflux2[i][69] for i in 500:1000])) < 0.1 * p[1]
# @test abs(p - mean([fhsampleslux2[i][69] for i in 500:1000])) < 0.2 * p[1]

# # @test abs(p - mean([fhsamplesflux2[i][70] for i in 500:1000])) < 0.1 * p[2]
# # @test abs(p - mean([fhsampleslux2[i][70] for i in 500:1000])) < 0.2 * p[2]

# # @test abs(p - mean([fhsamplesflux2[i][71] for i in 500:1000])) < 0.1 * p[3]
# # @test abs(p - mean([fhsampleslux2[i][71] for i in 500:1000])) < 0.2 * p[3]

# # @test abs(p - mean([fhsamplesflux2[i][72] for i in 500:1000])) < 0.1 * p[4]
# # @test abs(p - mean([fhsampleslux2[i][72] for i in 500:1000])) < 0.2 * p[4]

# # fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainlux1,
# #                                                               dataset = dataset,
# #                                                               draw_samples = 1000,
# #                                                               l2std = [0.05, 0.05],
# #                                                               phystd = [0.05, 0.05],
# #                                                               priorsNNw = (0.0, 3.0),
# #                                                               param = [
# #                                                                   Normal(1.5, 0.5),
# #                                                                   Normal(1.2, 0.5),
# #                                                                   Normal(3.3, 0.5),
# #                                                                   Normal(1.4, 0.5),
# #                                                               ], autodiff = true)

# # fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainlux1,
# #                                                               dataset = dataset,
# #                                                               draw_samples = 1000,
# #                                                               l2std = [0.05, 0.05],
# #                                                               phystd = [0.05, 0.05],
# #                                                               priorsNNw = (0.0, 3.0),
# #                                                               param = [
# #                                                                   Normal(1.5, 0.5),
# #                                                                   Normal(1.2, 0.5),
# #                                                                   Normal(3.3, 0.5),
# #                                                                   Normal(1.4, 0.5),
# #                                                               ], nchains = 2)

# # NOTES (WILL CLEAR LATER)
# # --------------------------------------------------------------------------------------------
# # Hamiltonian energy must be lowest(more paramters the better is it to map onto them)
# # full better than L2 and phy individual(test)
# # in mergephys more points after training points is better from 20->40
# # does consecutive runs bceome better? why?(plot 172)(same chain maybe)
# # does density of points in timespan matter dataset vs internal timespan?(plot 172)(100+0.01)
# # when training from 0-1 and phys from 1-5 with 1/150 simple nn slow,but bigger nn faster decrease in Hmailtonian
# # bigger time interval more curves to adapt to only more parameters adapt to that, better NN architecture
# # higher order logproblems solve better
# # repl up up are same instances? but reexecute calls are new?

# #Compare results against paper example
# # Lux chains support (DONE)
# # fix predictions for odes depending upon 1,p in f(u,p,t)(DONE)
# # lotka volterra learn curve beyond l2 losses(L2 losses determine accuracy of parameters)(parameters cant run free ∴ L2 interval only)
# # check if prameters estimation works(YES)
# # lotka volterra parameters estimate (DONE)

# using NeuralPDE, Lux, Flux, Optimization, OptimizationOptimJL
# import ModelingToolkit: Interval
# using Plots, StatsPlots
# plotly()
# # Profile.init()

# @parameters x y
# @variables u(..)
# Dxx = Differential(x)^2
# Dyy = Differential(y)^2

# # 2D PDE
# eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# # Boundary conditions
# bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
#     u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
# # Space and time domains
# domains = [x ∈ Interval(0.0, 1.0),
#     y ∈ Interval(0.0, 1.0)]

# # Neural network
# dim = 2 # number of dimensions
# chain = Flux.Chain(Flux.Dense(dim, 16, Lux.σ), Flux.Dense(16, 16, Lux.σ), Flux.Dense(16, 1))
# θ, re = destructure(chain)
# # Discretization
# dx = 0.05
# discretization = PhysicsInformedNN(chain, GridTraining(dx))

# @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

# pinnrep = symbolic_discretize(pde_system, discretization)
# typeof(pinnrep.phi)
# typeof(pinnrep.phi)
# typeof(re)
# pinnrep.phi([1, 2], θ)

# typeof(θ)

# print(pinnrep)
# pinnrep.eqs
# pinnrep.bcs
# pinnrep.domains
# pinnrep.eq_params
# pinnrep.defaults
# print(pinnrep.default_p)
# pinnrep.param_estim
# print(pinnrep.additional_loss)
# pinnrep.adaloss
# pinnrep.depvars
# pinnrep.indvars
# pinnrep.dict_depvar_input
# pinnrep.dict_depvars
# pinnrep.dict_indvars
# print(pinnrep.logger)
# pinnrep.multioutput
# pinnrep.iteration
# pinnrep.init_params
# pinnrep.flat_init_params
# pinnrep.phi
# pinnrep.derivative
# pinnrep.strategy
# pinnrep.pde_indvars
# pinnrep.bc_indvars
# pinnrep.pde_integration_vars
# pinnrep.bc_integration_vars
# pinnrep.integral
# pinnrep.symbolic_pde_loss_functions
# pinnrep.symbolic_bc_loss_functions
# pinnrep.loss_functions

# #  = discretize(pde_system, discretization)
# prob = symbolic_discretize(pde_system, discretization)
# # "The boundary condition loss functions"
# sum([prob.loss_functions.bc_loss_functions[i](θ) for i in eachindex(1:4)])
# sum([prob.loss_functions.pde_loss_functions[i](θ) for i in eachindex(1)])

# prob.loss_functions.full_loss_function(θ, 32)

# prob.loss_functions.bc_loss_functions[1](θ)

# prob.loss_functions.bc_loss_functions
# prob.loss_functions.full_loss_function
# prob.loss_functions.additional_loss_function
# prob.loss_functions.pde_loss_functions

# 1.3953060473003345 + 1.378102161087438 + 1.395376727128639 + 1.3783868705075002 +
# 0.22674532775196876
# # "The PDE loss functions"
# prob.loss_functions.pde_loss_functions
# prob.loss_functions.pde_loss_functions[1](θ)
# # "The full loss function, combining the PDE and boundary condition loss functions.This is the loss function that is used by the optimizer."
# prob.loss_functions.full_loss_function(θ, nothing)
# prob.loss_functions.full_loss_function(θ, 423423)

# # "The wrapped `additional_loss`, as pieced together for the optimizer."
# prob.loss_functions.additional_loss_function
# # "The pre-data version of the PDE loss function"
# prob.loss_functions.datafree_pde_loss_functions
# # "The pre-data version of the BC loss function"
# prob.loss_functions.datafree_bc_loss_functions

# using Random
# θ, st = Lux.setup(Random.default_rng(), chain)
# #Optimizer
# opt = OptimizationOptimJL.BFGS()

# #Callback function
# callback = function (p, l)
#     println("Current loss is: $l")
#     return false
# end

# res = Optimization.solve(prob, opt, callback = callback, maxiters = 1000)
# phi = discretization.phi

# # ------------------------------------------------
# using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
#       Plots
# import ModelingToolkit: Interval, infimum, supremum
# @parameters t, σ_, β, ρ
# @variables x(..), y(..), z(..)
# Dt = Differential(t)
# eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
#     Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
#     Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

# bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
# domains = [t ∈ Interval(0.0, 1.0)]
# dt = 0.01

# input_ = length(domains)
# n = 8
# chain1 = Lux.Chain(Lux.Dense(input_, n, Lux.σ), Lux.Dense(n, n, Lux.σ),
#                    Lux.Dense(n, n, Lux.σ),
#                    Lux.Dense(n, 1))
# chain2 = Lux.Chain(Lux.Dense(input_, n, Lux.σ), Lux.Dense(n, n, Lux.σ),
#                    Lux.Dense(n, n, Lux.σ),
#                    Lux.Dense(n, 1))
# chain3 = Lux.Chain(Lux.Dense(input_, n, Lux.σ), Lux.Dense(n, n, Lux.σ),
#                    Lux.Dense(n, n, Lux.σ),
#                    Lux.Dense(n, 1))

# function lorenz!(du, u, p, t)
#     du[1] = 10.0 * (u[2] - u[1])
#     du[2] = u[1] * (28.0 - u[3]) - u[2]
#     du[3] = u[1] * u[2] - (8 / 3) * u[3]
# end

# u0 = [1.0; 0.0; 0.0]
# tspan = (0.0, 1.0)
# prob = ODEProblem(lorenz!, u0, tspan)
# sol = solve(prob, Tsit5(), dt = 0.1)
# ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]
# function getData(sol)
#     data = []
#     us = hcat(sol(ts).u...)
#     ts_ = hcat(sol(ts).t...)
#     return [us, ts_]
# end
# data = getData(sol)

# (u_, t_) = data
# len = length(data[2])

# depvars = [:x, :y, :z]
# function additional_loss(phi, θ, p)
#     return sum(sum(abs2, phi[i](t_, θ[depvars[i]]) .- u_[[i], :]) / len for i in 1:1:3)
# end

# discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3],
#                                              NeuralPDE.GridTraining(dt),
#                                              param_estim = false,
#                                              additional_loss = additional_loss)
# @named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t), z(t)], [σ_, ρ, β],
#                               defaults = Dict([p .=> 1.0 for p in [σ_, ρ, β]]))
# prob = NeuralPDE.discretize(pde_system, discretization)
# callback = function (p, l)
#     println("Current loss is: $l")
#     return false
# end
# res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 5000)
# p_ = res.u[(end - 2):end] # p_ = [9.93, 28.002, 2.667]

# minimizers = [res.u.depvar[depvars[i]] for i in 1:3]
# ts = [infimum(d.domain):(dt / 10):supremum(d.domain) for d in domains][1]
# u_predict = [[discretization.phi[i]([t], minimizers[i])[1] for t in ts] for i in 1:3]
# plot(sol)
# plot!(ts, u_predict, label = ["x(t)" "y(t)" "z(t)"])

# discretization.multioutput
# discretization.chain
# discretization.strategy
# discretization.init_params
# discretization.phi
# discretization.derivative
# discretization.param_estim
# discretization.additional_loss
# discretization.adaptive_loss
# discretization.logger
# discretization.log_options
# discretization.iteration
# discretization.self_increment
# discretization.multioutput
# discretization.kwargs

# struct BNNODE1{P <: Vector{<:Distribution}}
#     chain::Any
#     Kernel::Any
#     draw_samples::UInt32
#     priorsNNw::Tuple{Float64, Float64}
#     param::P
#     l2std::Vector{Float64}
#     phystd::Vector{Float64}

#     function BNNODE1(chain, Kernel; draw_samples = 2000, priorsNNw = (0.0, 3.0), param = [],
#                      l2std = [0.05], phystd = [0.05])
#         BNNODE1(chain, Kernel, draw_samples, priorsNNw, param, l2std, phystd)
#     end
# end

# struct BNNODE3{C, K, P <: Union{Any, Vector{<:Distribution}}}
#     chain::C
#     Kernel::K
#     draw_samples::UInt32
#     priorsNNw::Tuple{Float64, Float64}
#     param::P
#     l2std::Vector{Float64}
#     phystd::Vector{Float64}

#     function BNNODE3(chain, Kernel; draw_samples = 2000, priorsNNw = (0.0, 3.0), param = [],
#                      l2std = [0.05], phystd = [0.05])
#         new{typeof(chain), typeof(Kernel), typeof(param)}(chain, Kernel, draw_samples,
#                                                           priorsNNw, param, l2std, phystd)
#     end
# end
# linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
# linear = (u, p, t) -> cos(2 * π * t)
# tspan = (0.0, 2.0)
# u0 = 0.0
# prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# ta = range(tspan[1], tspan[2], length = 300)
# u = [linear_analytic(u0, nothing, ti) for ti in ta]
# sol1 = solve(prob, Tsit5())

# # BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
# x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂[1:100], time[1:100]]

# # Call BPINN, create chain
# chainflux = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1))
# chainlux = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
# HMC
# solve(prob, BNNODE(chainflux, HMC))
# BNNODE1(chainflux, HMC, 2000)

# draw_samples = 2000
# priorsNNw = (0.0, 3.0)
# param = []
# l2std = [0.05]
# phystd = [0.05]
# @time BNNODE3(chainflux, HMC, draw_samples = 2000, priorsNNw = (0.0, 3.0),
#               param = [nothing],
#               l2std = [0.05], phystd = [0.05])
# typeof(Nothing) <: Vector{<:Distribution}
# Nothing <: Distribution
# {UnionAll} <: Distribution
# @time [Nothing]
# typeof([Nothing])
# @time [1]

# function test1(sum; c = 23, d = 32)
#     return sum + c + d
# end
# function test(a, b; c, d)
#     return test1(a + b, c, d)
# end

# test(2, 2)

# struct BNNODE3{C, K, P <: Union{Vector{Nothing}, Vector{<:Distribution}}}
#     chain::C
#     Kernel::K
#     draw_samples::Int64
#     priorsNNw::Tuple{Float64, Float64}
#     param::P
#     l2std::Vector{Float64}
#     phystd::Vector{Float64}

#     function BNNODE3(chain, Kernel; draw_samples,
#                      priorsNNw, param = [nothing], l2std, phystd)
#         new{typeof(chain), typeof(Kernel), typeof(param)}(chain,
#                                                           Kernel,
#                                                           draw_samples,
#                                                           priorsNNw,
#                                                           param, l2std,
#                                                           phystd)
#     end
# end

# function solve1(prob::DiffEqBase.AbstractODEProblem, alg::BNNODE3;
#                 dataset = [nothing], dt = 1 / 20.0,
#                 init_params = nothing, nchains = 1,
#                 autodiff = false, Integrator = Leapfrog,
#                 Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
#                 Metric = DiagEuclideanMetric, jitter_rate = 3.0,
#                 tempering_rate = 3.0, max_depth = 10, Δ_max = 1000,
#                 n_leapfrog = 10, δ = 0.65, λ = 0.3, progress = true,
#                 verbose = false)
#     chain = alg.chain
#     l2std = alg.l2std
#     phystd = alg.phystd
#     priorsNNw = alg.priorsNNw
#     Kernel = alg.Kernel
#     draw_samples = alg.draw_samples

#     param = alg.param == [nothing] ? [] : alg.param
#     mcmcchain, samples, statistics = ahmc_bayesian_pinn_ode(prob, chain, dataset = dataset,
#                                                             draw_samples = draw_samples,
#                                                             init_params = init_params,
#                                                             physdt = dt, l2std = l2std,
#                                                             phystd = phystd,
#                                                             priorsNNw = priorsNNw,
#                                                             param = param,
#                                                             nchains = nchains,
#                                                             autodiff = autodiff,
#                                                             Kernel = Kernel,
#                                                             Integrator = Integrator,
#                                                             Adaptor = Adaptor,
#                                                             targetacceptancerate = targetacceptancerate,
#                                                             Metric = Metric,
#                                                             jitter_rate = jitter_rate,
#                                                             tempering_rate = tempering_rate,
#                                                             max_depth = max_depth,
#                                                             Δ_max = Δ_max,
#                                                             n_leapfrog = n_leapfrog, δ = δ,
#                                                             λ = λ, progress = progress,
#                                                             verbose = verbose)
# end

# linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
# linear = (u, p, t) -> cos(2 * π * t)
# tspan = (0.0, 2.0)
# u0 = 0.0
# prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# ta = range(tspan[1], tspan[2], length = 300)
# u = [linear_analytic(u0, nothing, ti) for ti in ta]
# # sol1 = solve(prob, Tsit5())

# # BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
# x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
# time = vec(collect(Float64, ta))
# dataset = [x̂[1:100], time[1:100]]

# # Call BPINN, create chain
# chainflux = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1))
# chainlux = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
# HMC

# solve1(prob, a)
# a = BNNODE3(chainflux, HMC, draw_samples = 2000,
#             priorsNNw = (0.0, 3.0),
#             l2std = [0.05], phystd = [0.05])

# Define Lotka-Volterra model.
function lotka_volterra1(u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    dx = (α - β * y) * x # prey
    dy = (δ * x - γ) * y # predator

    return [dx, dy]
end

u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 6.0)
prob = ODEProblem(lotka_volterra1, u0, tspan, p)
solution = solve(prob, Tsit5(); saveat = 0.05)

as = reduce(hcat, solution.u)
as[1, :]
# Plot simulation.
time = solution.t
u = hcat(solution.u...)
# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x = u[1, :] + 0.5 * randn(length(u[1, :]))
y = u[2, :] + 0.5 * randn(length(u[1, :]))
dataset = [x[1:50], y[1:50], time[1:50]]
# scatter!(time, [x, y])
# scatter!(dataset[3], [dataset[2], dataset[1]])

# NN has 2 outputs as u -> [dx,dy]
chainlux1 = Lux.Chain(Lux.Dense(1, 6, Lux.tanh), Lux.Dense(6, 6, Lux.tanh),
    Lux.Dense(6, 2))
chainflux1 = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh), Flux.Dense(6, 2))

fh_mcmc_chainflux1, fhsamplesflux1, fhstatsflux1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
    dataset = dataset,
    draw_samples = 1000,
    l2std = [
        0.05,
        0.05,
    ],
    phystd = [
        0.05,
        0.05,
    ],
    priorsNNw = (0.0, 3.0), progress = true)
ahmc_bayesian_pinn_ode(prob, chainflux1,
    dataset = dataset,
    draw_samples = 1000,
    l2std = [
        0.05,
        0.05,
    ],
    phystd = [
        0.05,
        0.05,
    ],
    priorsNNw = (0.0, 3.0), progress = true)

#     2×171 Matrix{Float64}:
#  -0.5  -0.518956  -0.529639  …  -1.00266  -1.01049
#   2.0   1.97109    1.92747       0.42619   0.396335

#     2-element Vector{Float64}:
#  -119451.94949911036
#  -128543.23714618056

# alg = NeuralPDE.BNNODE(chainflux1,
#     dataset = dataset,
#     draw_samples = 1000,
#     l2std = [
#         0.05,
#         0.05,
#     ],
#     phystd = [
#         0.05,
#         0.05,
#     ],
#     priorsNNw = (0.0,
#         3.0),
#     param = [
#         Normal(4.5,
#             5),
#         Normal(7,
#             2),
#         Normal(5,
#             2),
#         Normal(-4,
#             6),
#     ],
#     n_leapfrog = 30, progress = true)

# sol3flux_pestim = solve(prob, alg)

#  ----------------------------------------------
# original paper implementation
# 25 points 
run1  #7.70593 Particles{Float64, 1}
run2 #6.66347 Particles{Float64, 1} 
run3 #6.84827 Particles{Float64, 1} 

# 50 points 
run1 #7.83577 Particles{Float64, 1}
run2 #6.49477 Particles{Float64, 1}
run3 #6.47421 Particles{Float64, 1}

# 100 points 
run1 #5.96604 Particles{Float64, 1}
run2 #6.05432 Particles{Float64, 1}
run3 #6.08856 Particles{Float64, 1}

# Full likelihood(uses total variation regularized differentiation) 
# 25 points 
run1 #6.41722 Particles{Float64, 1}
run2 #6.42782 Particles{Float64, 1}
run3 #6.42782 Particles{Float64, 1}

# 50 points
run1 #5.71268 Particles{Float64, 1}
run2 #5.74599 Particles{Float64, 1}
run3 #5.74599 Particles{Float64, 1}

# 100 points  
run1 #6.59097 Particles{Float64, 1}
run2 #6.62813 Particles{Float64, 1}
run3 #6.62813 Particles{Float64, 1}

using Plots, StatsPlots
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
tspan = (0.0, 6.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Plot simulation.

solution = solve(prob, Tsit5(); saveat = 0.05)
plot(solve(prob, Tsit5()))

# Dataset creation for parameter estimation
time = solution.t
u = hcat(solution.u...)
x = u[1, :] + 0.5 * randn(length(u[1, :]))
y = u[2, :] + 0.5 * randn(length(u[1, :]))
dataset = [x, y, time]

# Neural Networks must have 2 outputs as u -> [dx,dy] in function lotka_volterra()
chainflux = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh), Flux.Dense(6, 2)) |>
            Flux.f64

chainlux = Lux.Chain(Lux.Dense(1, 6, Lux.tanh), Lux.Dense(6, 6, Lux.tanh), Lux.Dense(6, 2))

alg1 = NeuralPDE.BNNODE(chainflux,
    dataset = dataset,
    draw_samples = 1000,
    l2std = [
        0.01,
        0.01,
    ],
    phystd = [
        0.01,
        0.01,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        LogNormal(1.5,
            0.5),
        LogNormal(1.2,
            0.5),
        LogNormal(3.3,
            1),
        LogNormal(1.4,
            1)],
    n_leapfrog = 30, progress = true)

sol_flux_pestim = solve(prob, alg1)

# Dataset not needed as we are solving the equation with ideal parameters
alg2 = NeuralPDE.BNNODE(chainlux,
    draw_samples = 1000,
    l2std = [
        0.05,
        0.05,
    ],
    phystd = [
        0.05,
        0.05,
    ],
    priorsNNw = (0.0,
        3.0),
    n_leapfrog = 30, progress = true)

sol_lux = solve(prob, alg2)

#testing timepoints must match keyword arg `saveat`` timepoints of solve() call
t = collect(Float64, prob.tspan[1]:(1 / 50.0):prob.tspan[2])

# plotting solution for x,y for chain_flux
plot(t, sol_flux_pestim.ensemblesol[1])
plot!(t, sol_flux_pestim.ensemblesol[2])

plot(sol_flux_pestim.ens1mblesol[1])
plot!(sol_flux_pestim.ensemblesol[2])

# estimated ODE parameters by .estimated_ode_params, weights and biases by .estimated_nn_params
sol_flux_pestim.estimated_nn_params
sol_flux_pestim.estimated_ode_params

# plotting solution for x,y for chain_lux
plot(t, sol_lux.ensemblesol[1])
plot!(t, sol_lux.ensemblesol[2])

# estimated weights and biases by .estimated_nn_params for chain_lux
sol_lux.estimated_nn_params

# # ----------------------------------stats-----------------------------
# #   ----------------------------
# # -----------------------------
# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -3118.0639515039957
# Sampling 100%|███████████████████████████████| Time: 0:04:47 

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -3118.0639515039957
# Sampling 100%|███████████████████████████████| Time: 0:03:38 

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -3118.0639515039957
# Sampling 100%|███████████████████████████████| Time: 0:04:12 
# #   --------------------------
# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -6242.351071278482
# Sampling 100%|███████████████████████████████| Time: 0:05:09 

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -6242.351071278482
# Sampling 100%|███████████████████████████████| Time: 0:04:47 

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -6242.351071278482
# Sampling 100%|███████████████████████████████| Time: 0:04:25 
# #   --------------
# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -12497.32657780532
# Sampling 100%|███████████████████████████████| Time: 0:06:47 

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -12497.32657780532
# Sampling 100%|███████████████████████████████| Time: 0:05:54

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2loss2 Logpdf is : -12497.32657780532
# Sampling 100%|███████████████████████████████| Time: 0:05:46
# # ------------------------
# # -----------------------
# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -882.2934218498742
# L2loss2 Logpdf is : -3118.0639515039957
# Sampling 100%|███████████████████████████████| Time: 0:04:06

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -882.2934218498742
# L2loss2 Logpdf is : -3118.0639515039957
# Sampling 100%|███████████████████████████████| Time: 0:03:32

# physics Logpdf is : -15740.509286661572
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -882.2934218498742
# L2loss2 Logpdf is : -3118.0639515039957
# Sampling 100%|███████████████████████████████| Time: 0:03:01 
# # --------------------------
# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1411.1717435511828
# L2loss2 Logpdf is : -6242.351071278482
# Sampling 100%|███████████████████████████████| Time: 0:04:02

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1411.1717435511828
# L2loss2 Logpdf is : -6242.351071278482
# Sampling 100%|███████████████████████████████| Time: 0:04:08

# physics Logpdf is : -18864.79640643607
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -1411.1717435511828
# L2loss2 Logpdf is : -6242.351071278482
# Sampling 100%|███████████████████████████████| Time: 0:04:15
# # ----------------------------
# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -3240.067149411982
# L2loss2 Logpdf is : -12497.32657780532
# Sampling 100%|███████████████████████████████| Time: 0:05:37

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -3240.067149411982
# L2loss2 Logpdf is : -12497.32657780532
# Sampling 100%|███████████████████████████████| Time: 0:06:02

# physics Logpdf is : -25119.77191296288
# prior Logpdf is : -139.5069300318621
# L2lossData Logpdf is : -3240.067149411982
# L2loss2 Logpdf is : -12497.32657780532
# Sampling 100%|███████████████████████████████| Time: 0:06:13

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

using NeuralPDE, Flux, OptimizationOptimisers

function diffeq(u, p, t)
    u1, u2 = u
    return [u2, p[1] + p[2] * sin(u1) + p[3] * u2]
end
p = [5, -10, -1.7]
u0 = [-1.0, 7.0]
tspan = (0.0, 10.0)
prob = ODEProblem(ODEFunction(diffeq), u0, tspan, p)

chainnew = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 8, tanh), Flux.Dense(8, 2)) |>
           Flux.f64

opt = OptimizationOptimisers.Adam(0.1)
opt = Optimisers.ADAGrad(0.1)
opt = Optimisers.AdaMax(0.01)
algnew = NeuralPDE.NNODE(chainnew, opt)
solution_new = solve(prob, algnew, verbose = true,
    abstol = 1e-10, maxiters = 7000)
u = reduce(hcat, solution_new.u)
plot(solution_new.t, u[1, :])
plot!(solution_new.t, u[2, :])

algnew = NeuralPDE.BNNODE(chainnew, draw_samples = 200,
    n_leapfrog = 30, progress = true)
solution_new = solve(prob, algnew)

@parameters t
@variables u1(..), u2(..)
D = Differential(t)
eq = [D(u1(t)) ~ u2(t),
    D(u2(t)) ~ 5 - 10 * sin(u1(t)) - 1.7 * u2(t)];

import ModelingToolkit: Interval
bcs = [u1(0) ~ -1, u2(0) ~ 7]
domains = [t ∈ Interval(0.0, 10.0)]
dt = 0.01

input_ = length(domains) # number of dimensions
n = 16
chain = [Lux.Chain(Lux.Dense(input_, n, Lux.σ), Lux.Dense(n, n, Lux.σ), Lux.Dense(n, 1))
         for _ in 1:2]

@named pde_system = PDESystem(eq, bcs, domains, [t], [u1(t), u2(t)])

strategy = NeuralPDE.GridTraining(dt)
discretization = PhysicsInformedNN(chain, strategy)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    println("loss: ", l)
    # println("pde_losses: ", map(l_ -> l_(p), pde_loss_functions))
    # println("bcs_losses: ", map(l_ -> l_(p), bc_loss_functions))
    return false
end

loss_functions = [pde_loss_functions; bc_loss_functions]

function loss_function(θ, p)
    sum(map(l -> l(θ), loss_functions))
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, sym_prob.flat_init_params)

res = Optimization.solve(prob,
    OptimizationOptimJL.BFGS();
    callback = callback,
    maxiters = 1000)
phi = discretization.phi