# # Testing Code
using Test, MCMCChains
using ForwardDiff, Distributions, OrdinaryDiffEq
using OptimizationOptimisers, AdvancedHMC, Lux
using Statistics, Random, Functors, ComponentArrays
using NeuralPDE, MonteCarloMeasurements
using Flux

# note that current testing bounds can be easily further tightened but have been inflated for support for Julia build v1
# on latest Julia version it performs much better for below tests
Random.seed!(100)

@testset "Example 1 - without parameter estimation" begin
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

    chainlux = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux)

    fh_mcmc_chain, fhsamples, fhstats = ahmc_bayesian_pinn_ode(
        prob, chainlux, draw_samples = 2500)

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
    @test mean(abs.(x̂ .- meanscurve)) < 0.05
    @test mean(abs.(physsol1 .- meanscurve)) < 0.005

    #--------------------- solve() call
    @test mean(abs.(x̂1 .- pmean(sol1lux.ensemblesol[1]))) < 0.025
    @test mean(abs.(physsol0_1 .- pmean(sol1lux.ensemblesol[1]))) < 0.025
end

@testset "Example 2 - with parameter estimation" begin
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

    chainlux1 = Lux.Chain(Lux.Dense(1, 7, tanh), Lux.Dense(7, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux1)

    fh_mcmc_chain, fhsamples, fhstats = ahmc_bayesian_pinn_ode(prob, chainlux1,
        dataset = dataset,
        draw_samples = 2500,
        physdt = 1 / 50.0,
        priorsNNw = (0.0, 3.0),
        param = [LogNormal(9, 0.5)])

    alg = BNNODE(chainlux1, dataset = dataset,
        draw_samples = 2500,
        physdt = 1 / 50.0,
        priorsNNw = (0.0,
            3.0),
        param = [
            LogNormal(9,
            0.5)
        ])

    sol2lux = solve(prob, alg)

    # testing points
    t = time
    # Mean of last 500 sampled parameter's curves(flux and lux chains)[Ensemble predictions]
    θ = [vector_to_parameters(fhsamples[i][1:(end - 1)], θinit)
         for i in 2000:length(fhsamples)]
    luxar = [chainlux1(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    # --------------------- ahmc_bayesian_pinn_ode() call
    @test mean(abs.(physsol1 .- meanscurve)) < 0.15

    # ESTIMATED ODE PARAMETERS (NN1 AND NN2)
    @test abs(p - mean([fhsamples[i][23] for i in 2000:length(fhsamples)])) < abs(0.35 * p)

    #-------------------------- solve() call
    @test mean(abs.(physsol1_1 .- pmean(sol2lux.ensemblesol[1]))) < 8e-2

    # ESTIMATED ODE PARAMETERS (NN1 AND NN2)
    @test abs(p - sol2lux.estimated_de_params[1]) < abs(0.15 * p)
end

@testset "Example 3" begin
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
    physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

    # seperate set of points for testing the solve() call (it uses saveat 1/50 hence here length 501)
    time1 = vec(collect(Float64, range(tspan[1], tspan[2], length = 501)))
    physsol2 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

    chainlux12 = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux12)

    fh_mcmc_chainlux12, fhsampleslux12, fhstatslux12 = ahmc_bayesian_pinn_ode(
        prob, chainlux12,
        draw_samples = 1500,
        l2std = [0.03],
        phystd = [0.03],
        priorsNNw = (0.0,
            10.0))

    fh_mcmc_chainlux22, fhsampleslux22, fhstatslux22 = ahmc_bayesian_pinn_ode(
        prob, chainlux12,
        dataset = dataset,
        draw_samples = 1500,
        l2std = [0.03],
        phystd = [0.03],
        priorsNNw = (0.0,
            10.0),
        param = [
            Normal(-7,
            4)
        ])

    alg = BNNODE(chainlux12,
        dataset = dataset,
        draw_samples = 1500,
        l2std = [0.03],
        phystd = [0.03],
        priorsNNw = (0.0,
            10.0),
        param = [
            Normal(-7,
            4)
        ])

    sol3lux_pestim = solve(prob, alg)

    # testing timepoints
    t = sol.t
    #------------------------------ ahmc_bayesian_pinn_ode() call
    # Mean of last 500 sampled parameter's curves(lux chains)[Ensemble predictions]
    θ = [vector_to_parameters(fhsampleslux12[i], θinit)
         for i in 1000:length(fhsampleslux12)]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    θ = [vector_to_parameters(fhsampleslux22[i][1:(end - 1)], θinit)
         for i in 1000:length(fhsampleslux22)]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    @test mean(abs.(sol.u .- meanscurve2_1)) < 1e-1
    @test mean(abs.(physsol1 .- meanscurve2_1)) < 1e-1
    @test mean(abs.(sol.u .- meanscurve2_2)) < 5e-2
    @test mean(abs.(physsol1 .- meanscurve2_2)) < 5e-2

    # estimated parameters(lux chain)
    param1 = mean(i[62] for i in fhsampleslux22[1000:length(fhsampleslux22)])
    @test abs(param1 - p) < abs(0.3 * p)

    #-------------------------- solve() call
    # (lux chain)
    @test mean(abs.(physsol2 .- pmean(sol3lux_pestim.ensemblesol[1]))) < 0.15
    # estimated parameters(lux chain)
    param1 = sol3lux_pestim.estimated_de_params[1]
    @test abs(param1 - p) < abs(0.45 * p)
end

@testset "Translating from Flux" begin
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
    fh_mcmc_chain, fhsamples, fhstats = ahmc_bayesian_pinn_ode(
        prob, chainflux, draw_samples = 2500)
    alg = BNNODE(chainflux, draw_samples = 2500)
    @test alg.chain isa Lux.AbstractLuxLayer
end

@testset "Example 3 but with the new objective" begin
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
    x̂ = u .+ (0.3 .* randn(size(u)))
    dataset = [x̂, time]
    physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

    # seperate set of points for testing the solve() call (it uses saveat 1/50 hence here length 501)
    time1 = vec(collect(Float64, range(tspan[1], tspan[2], length = 501)))
    physsol2 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

    chainlux12 = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux12)

    fh_mcmc_chainlux12, fhsampleslux12, fhstatslux12 = ahmc_bayesian_pinn_ode(
        prob, chainlux12,
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.1],
        phystd = [0.03],
        priorsNNw = (0.0,
            1.0),
        param = [
            Normal(-7, 3)
        ])

    fh_mcmc_chainlux22, fhsampleslux22, fhstatslux22 = ahmc_bayesian_pinn_ode(
        prob, chainlux12,
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.1],
        phystd = [0.03],
        priorsNNw = (0.0,
            1.0),
        param = [
            Normal(-7, 3)
        ], estim_collocate = true)

    alg = BNNODE(chainlux12,
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.1],
        phystd = [0.03],
        priorsNNw = (0.0,
            1.0),
        param = [
            Normal(-7, 3)
        ], estim_collocate = true)

    sol3lux_pestim = solve(prob, alg)

    # testing timepoints
    t = sol.t
    #------------------------------ ahmc_bayesian_pinn_ode() call
    # Mean of last 500 sampled parameter's curves(lux chains)[Ensemble predictions]
    θ = [vector_to_parameters(fhsampleslux12[i][1:(end - 1)], θinit)
         for i in 750:length(fhsampleslux12)]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    θ = [vector_to_parameters(fhsampleslux22[i][1:(end - 1)], θinit)
         for i in 750:length(fhsampleslux22)]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    @test mean(abs.(sol.u .- meanscurve2_2)) < 6e-2
    @test mean(abs.(physsol1 .- meanscurve2_2)) < 6e-2
    @test mean(abs.(sol.u .- meanscurve2_1)) > mean(abs.(sol.u .- meanscurve2_2))
    @test mean(abs.(physsol1 .- meanscurve2_1)) > mean(abs.(physsol1 .- meanscurve2_2))

    # estimated parameters(lux chain)
    param2 = mean(i[62] for i in fhsampleslux22[750:length(fhsampleslux22)])
    @test abs(param2 - p) < abs(0.25 * p)

    param1 = mean(i[62] for i in fhsampleslux12[750:length(fhsampleslux12)])
    @test abs(param1 - p) < abs(0.75 * p)
    @test abs(param2 - p) < abs(param1 - p)

    #-------------------------- solve() call
    # (lux chain)
    @test mean(abs.(physsol2 .- pmean(sol3lux_pestim.ensemblesol[1]))) < 0.1
    # estimated parameters(lux chain)
    param3 = sol3lux_pestim.estimated_de_params[1]
    @test abs(param3 - p) < abs(0.2 * p)
end

@testset "Example 4 - improvement" begin
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
    tspan = (0.0, 4.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)

    # Solve using OrdinaryDiffEq.jl solver
    dt = 0.2
    solution = solve(prob, Tsit5(); saveat = dt)

    times = solution.t
    u = hcat(solution.u...)
    x = u[1, :] + (0.8 .* randn(length(u[1, :])))
    y = u[2, :] + (0.8 .* randn(length(u[2, :])))
    dataset = [x, y, times]

    chain = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh),
        Lux.Dense(6, 2))

    alg1 = BNNODE(chain;
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.2, 0.2],
        phystd = [0.1, 0.1],
        priorsNNw = (0.0, 1.0),
        param = [
            Normal(2, 0.5),
            Normal(2, 0.5),
            Normal(2, 0.5),
            Normal(2, 0.5)])

    alg2 = BNNODE(chain;
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.2, 0.2],
        phystd = [0.1, 0.1],
        priorsNNw = (0.0, 1.0),
        param = [
            Normal(2, 0.5),
            Normal(2, 0.5),
            Normal(2, 0.5),
            Normal(2, 0.5)], estim_collocate = true)

    @time sol_pestim1 = solve(prob, alg1; saveat = dt)
    @time sol_pestim2 = solve(prob, alg2; saveat = dt)

    unsafe_comparisons(true)
    bitvec = abs.(p .- sol_pestim1.estimated_de_params) .>
             abs.(p .- sol_pestim2.estimated_de_params)
    @test bitvec == ones(size(bitvec))
end
