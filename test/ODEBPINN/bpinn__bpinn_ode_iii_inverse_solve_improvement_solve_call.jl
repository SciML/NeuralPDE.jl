using NeuralPDE
using Test

@testset "BPINN ODE III: Inverse solve Improvement solve call" begin
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
    x̂ = u .+ (0.1 .* randn(size(u)))

    # dx = 0.1 Gridtraining for newloss
    dataset = [x̂, time, ones(length(time))]

    # set of points for testing the solve() call (it uses saveat 1/50 hence here length 501)
    time1 = vec(collect(Float64, range(tspan[1], tspan[2], length = 501)))
    physsol2 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]

    chainlux12 = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
    θinit, st = Lux.setup(Random.default_rng(), chainlux12)

    alg = BNNODE(
        chainlux12,
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.1],
        phystd = [0.01],
        phynewstd = (p) -> [0.01],
        priorsNNw = (
            0.0,
            1.0,
        ),
        param = [
            Normal(-7, 3),
        ], numensemble = 200,
        estim_collocate = true
    )

    sol3lux_pestim = solve(prob, alg)

    #-------------------------- solve() call
    @test mean(abs.(physsol2 .- pmean(sol3lux_pestim.ensemblesol[1]))) < 1.0e-2

    # estimated parameters
    param3 = sol3lux_pestim.estimated_de_params[1]
    @test abs(param3 - p) < abs(0.05 * p)
end
