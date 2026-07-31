using NeuralPDE
using Test

@testset "BPINN ODE IV: Inverse solve Improvement" begin
    using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements
    using FastGaussQuadrature
    Random.seed!(100)

    function lotka_volterra(u, p, t)
        # Model parameters.
        α, δ = p
        # Current state.
        x, y = u

        # Evaluate differential equations.
        dx = (α - y) * x # prey
        dy = (x - δ) * y  # predator

        return [dx, dy]
    end

    # initial-value problem.
    u0 = [1.0, 1.0]
    p = [1.5, 3.0]
    tspan = (0.0, 4.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)

    N = 20
    # x, w = gausslegendre(N) # does not include endpoints
    x, w = gausslobatto(N)
    # x, w = clenshaw_curtis(N)
    a = tspan[1]
    b = tspan[2]
    # transform the roots and weights
    # x = map((x) -> (2 * (t - a) / (b - a)) - 1, x)
    t = map((x) -> (x * (b - a) + (b + a)) / 2, x)
    W = map((x) -> x * (b - a) / 2, w)
    solution = solve(prob, Tsit5(); saveat = t)
    times = solution.t
    u = hcat(solution.u...)
    x = u[1, :] + (0.5 .* randn(length(u[1, :])))
    y = u[2, :] + (0.5 .* randn(length(u[2, :])))
    dataset = [x, y, times, W]

    chain = Lux.Chain(
        Lux.Dense(1, 7, tanh), Lux.Dense(7, 7, tanh),
        Lux.Dense(7, 2)
    )

    alg1 = BNNODE(
        chain;
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.5, 0.5],
        phystd = [0.5, 0.5],
        priorsNNw = (0.0, 1.0),
        param = [
            Normal(-7, 2),
            Normal(-7, 2),
        ]
    )

    alg2 = BNNODE(
        chain;
        dataset = dataset,
        draw_samples = 1000,
        l2std = [0.5, 0.5],
        phystd = [0.5, 0.5],
        phynewstd = (p) -> [0.5, 0.5],
        priorsNNw = (0.0, 1.0),
        param = [
            Normal(-7, 2),
            Normal(-7, 2),
        ], estim_collocate = true
    )

    dt = 0.05
    @time sol_pestim1 = solve(prob, alg1; saveat = dt)
    @time sol_pestim2 = solve(prob, alg2; saveat = dt)
    # OrdinaryDiffEq.jl solve at sol.timepoints.
    solution = solve(prob, Tsit5(); saveat = dt)
    u = hcat(solution.u...)

    unsafe_comparisons(true)
    bitvec = abs.(p .- sol_pestim1.estimated_de_params) .>
        abs.(p .- sol_pestim2.estimated_de_params)
    @test bitvec == ones(size(bitvec))

    @test mean(abs, u[1, :] .- pmean(sol_pestim1.ensemblesol[1])) >
        mean(abs, u[1, :] .- pmean(sol_pestim2.ensemblesol[1]))
    @test mean(abs, u[2, :] .- pmean(sol_pestim1.ensemblesol[2])) >
        mean(abs, u[2, :] .- pmean(sol_pestim2.ensemblesol[2]))

    @test mean(abs2, u[1, :] .- pmean(sol_pestim2.ensemblesol[1])) < 1.0e-1
    @test mean(abs2, u[2, :] .- pmean(sol_pestim2.ensemblesol[2])) < 2.0e-2

    @test abs(sol_pestim2.estimated_de_params[1] - p[1]) < 0.05p[1]
    @test abs(sol_pestim2.estimated_de_params[2] - p[2]) < 0.1p[2]
end
