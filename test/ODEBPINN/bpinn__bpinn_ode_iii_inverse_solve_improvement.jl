using NeuralPDE
using Test

@testset "BPINN ODE III: Inverse solve Improvement" begin
    using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, ComponentArrays,
        MonteCarloMeasurements, FastGaussQuadrature
    Random.seed!(100)

    # (original Improvement tests can be run with 100 training points, check solve call tests.)
    # new model is always better (especially less points, more noise etc), given the correct std & enough samples.
    # std for the equation is limited ~ (var propagated via data points through chosen equation var/phystd)
    # for inverse problems ratio of datapoints and unsolved datapoints is important.

    N = 20  # choose number of nodes, enough to approximate 2n-2 degree polynomials (gauss-lobatto case)
    # x, w = gausslegendre(N) # does not include endpoints
    x, w = gausslobatto(N)
    # x, w = clenshaw_curtis(N)

    tspan = (0.0, 10.0)
    a = tspan[1]
    b = tspan[2]
    # transform the roots and weights
    # x = map((x) -> (2 * (t - a) / (b - a)) - 1, x)
    t = map((x) -> (x * (b - a) + (b + a)) / 2, x)
    W = map((x) -> x * (b - a) / 2, w)

    linear = (u, p, t) -> u / p + exp(t / p) * cos(t)
    u0 = 0.0
    p = -5.0
    prob = ODEProblem(linear, u0, tspan, p)
    linear_analytic = (u0, p, t) -> exp(t / p) * (u0 + sin(t))

    # SOLUTION AND CREATE DATASET
    sol = solve(prob, Tsit5(); saveat = t)
    u = sol.u  # use these points for collocation
    ts = sol.t

    # old model finds less noisy signal easier to learn. (i think its overfitting)
    x̂ = u .+ (0.1 .* randn(size(u)))
    dataset = [x̂, ts, W]
    physsol1 = [linear_analytic(prob.u0, p, ts[i]) for i in eachindex(ts)]
    chainlux12 = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux12)

    # you could always directly fit model to all data, but it ignores equation, overfits data.
    fh_mcmc_chainlux22, fhsampleslux22,
        fhstatslux22 = ahmc_bayesian_pinn_ode(
        prob, chainlux12,
        dataset = dataset,
        draw_samples = 2500,
        l2std = [0.1],
        phystd = [0.1],
        phynewstd = (p) -> [0.1 / p],
        priorsNNw = (
            0.0,
            1.0,
        ),
        param = [
            Normal(-7, 3),
        ], estim_collocate = true
    )

    fh_mcmc_chainlux12, fhsampleslux12,
        fhstatslux12 = ahmc_bayesian_pinn_ode(
        prob, chainlux12,
        dataset = dataset,
        draw_samples = 2500,
        l2std = [0.1],
        phystd = [0.1],
        priorsNNw = (
            0.0,
            1.0,
        ),
        param = [
            Normal(-7, 3),
        ]
    )

    # testing timepoints
    t = sol.t
    #------------------------------ ahmc_bayesian_pinn_ode() call
    # Mean of last 100 sampled parameter's curves(lux chains)[Ensemble predictions]
    θ = [
        vector_to_parameters(fhsampleslux12[i][1:(end - 1)], θinit)
            for i in 2400:length(fhsampleslux12)
    ]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_1 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    θ = [
        vector_to_parameters(fhsampleslux22[i][1:(end - 1)], θinit)
            for i in 2400:length(fhsampleslux22)
    ]
    luxar = [chainlux12(t', θ[i], st)[1] for i in eachindex(θ)]
    luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
    meanscurve2_2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean

    @test mean(abs.(sol.u .- meanscurve2_2)) < 5.0e-2
    @test mean(abs.(physsol1 .- meanscurve2_2)) < 5.0e-2
    @test mean(abs.(sol.u .- meanscurve2_1)) > mean(abs.(sol.u .- meanscurve2_2))
    @test mean(abs.(physsol1 .- meanscurve2_1)) > mean(abs.(physsol1 .- meanscurve2_2))

    param2 = mean(i[62] for i in fhsampleslux22[2400:length(fhsampleslux22)])
    @test abs(param2 - p) < abs(0.3 * p)

    param1 = mean(i[62] for i in fhsampleslux12[2400:length(fhsampleslux12)])
    @test abs(param1 - p) > abs(0.5 * p)
    @test abs(param2 - p) < abs(param1 - p)
end
