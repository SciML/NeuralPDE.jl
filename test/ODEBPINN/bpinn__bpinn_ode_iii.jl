using NeuralPDE
using Test

@testset "BPINN ODE III" begin
    using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements

    Random.seed!(100)

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

    # Note this is signal scaled gaussian noise, therefore the noise is biased and L2 penalizes high std points implicitly.
    x̂ = u .+ (u .* 0.1) .* randn(size(u))
    dataset = [x̂, time]
    physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

    # separate set of points for testing the solve() call (it uses saveat 1/50 hence here length 501)
    time1 = vec(collect(Float64, range(tspan[1], tspan[2], length = 501)))
    physsol2 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

    chainlux12 = Chain(Dense(1, 6, tanh), Dense(6, 6, tanh), Dense(6, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux12)

    # this a forward solve
    fh_mcmc_chainlux12, fhsampleslux12,
        fhstatslux12 = ahmc_bayesian_pinn_ode(
        prob, chainlux12, draw_samples = 500, phystd = [0.01], priorsNNw = (0.0, 10.0)
    )

    fh_mcmc_chainlux22, fhsampleslux22,
        fhstatslux22 = ahmc_bayesian_pinn_ode(
        prob, chainlux12, dataset = dataset, draw_samples = 500, l2std = [0.02],
        phystd = [0.05], priorsNNw = (0.0, 10.0), param = [Normal(-7, 4)]
    )

    alg = BNNODE(
        chainlux12, dataset = dataset, draw_samples = 500, l2std = [0.02],
        phystd = [0.05], priorsNNw = (0.0, 10.0), param = [Normal(-7, 4)]
    )

    sol3lux_pestim = solve(prob, alg)

    # testing timepoints
    t = sol.t
    #------------------------------ ahmc_bayesian_pinn_ode() call
    # Mean of last 500 sampled parameter's curves(lux chains)[Ensemble predictions]
    θ = [
        vector_to_parameters(fhsampleslux12[i], θinit)
            for i in 400:length(fhsampleslux12)
    ]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    θ = [
        vector_to_parameters(fhsampleslux22[i][1:(end - 1)], θinit)
            for i in 400:length(fhsampleslux22)
    ]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    @test mean(abs, sol.u .- meanscurve2_1) < 1.0e-2
    @test mean(abs, physsol1 .- meanscurve2_1) < 1.0e-2
    @test mean(abs, sol.u .- meanscurve2_2) < 1.5
    @test mean(abs, physsol1 .- meanscurve2_2) < 1.5

    # estimated parameters(lux chain)
    param1 = mean(i[62] for i in fhsampleslux22[400:length(fhsampleslux22)])
    @test abs(param1 - p) < abs(0.5 * p)
end
