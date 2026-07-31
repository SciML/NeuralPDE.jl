using NeuralPDE
using Test

@testset "BPINN ODE I: Without Param Estimation" begin
    using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements

    Random.seed!(100)

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

    chainlux = Chain(Dense(1, 7, tanh), Dense(7, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux)

    fh_mcmc_chain, fhsamples,
        fhstats = ahmc_bayesian_pinn_ode(
        prob, chainlux, draw_samples = 2500
    )

    alg = BNNODE(chainlux, draw_samples = 2500)
    sol1lux = solve(prob, alg)

    # testing points
    t = time
    # Mean of last 500 sampled parameter's curves[Ensemble predictions]
    θ = [vector_to_parameters(fhsamples[i], θinit) for i in 2000:length(fhsamples)]
    luxar = [chainlux(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    # --------------------- ahmc_bayesian_pinn_ode() call
    @test mean(abs.(x̂ .- meanscurve)) < 0.08
    @test mean(abs.(physsol1 .- meanscurve)) < 0.01

    #--------------------- solve() call
    @test mean(abs.(x̂1 .- pmean(sol1lux.ensemblesol[1]))) < 0.04
    @test mean(abs.(physsol0_1 .- pmean(sol1lux.ensemblesol[1]))) < 0.04
end
