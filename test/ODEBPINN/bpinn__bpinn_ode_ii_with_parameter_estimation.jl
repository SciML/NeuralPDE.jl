using NeuralPDE
using Test

@testset "BPINN ODE II: With Parameter Estimation" begin
    using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements

    Random.seed!(100)

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

    chainlux1 = Chain(Dense(1, 7, tanh), Dense(7, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux1)

    fh_mcmc_chain, fhsamples,
        fhstats = ahmc_bayesian_pinn_ode(
        prob, chainlux1, dataset = dataset, draw_samples = 2500,
        physdt = 1 / 50.0, priorsNNw = (0.0, 3.0), param = [LogNormal(9, 0.5)]
    )

    alg = BNNODE(
        chainlux1, dataset = dataset, draw_samples = 2500, physdt = 1 / 50.0,
        priorsNNw = (0.0, 3.0), param = [LogNormal(9, 0.5)]
    )

    sol2lux = solve(prob, alg)

    # testing points
    t = time
    # Mean of last 500 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
    θ = [
        vector_to_parameters(fhsamples[i][1:(end - 1)], θinit)
            for i in 2000:length(fhsamples)
    ]
    luxar = [chainlux1(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    # --------------------- ahmc_bayesian_pinn_ode() call
    @test mean(abs.(physsol1 .- meanscurve)) < 0.15

    # ESTIMATED ODE PARAMETERS (NN1 AND NN2)
    @test abs(p - mean([fhsamples[i][23] for i in 2000:length(fhsamples)])) < abs(0.35 * p)

    #-------------------------- solve() call
    @test mean(abs.(physsol1_1 .- pmean(sol2lux.ensemblesol[1]))) < 8.0e-2

    # ESTIMATED ODE PARAMETERS (NN1 AND NN2)
    @test abs(p - sol2lux.estimated_de_params[1]) < abs(0.15 * p)
end
