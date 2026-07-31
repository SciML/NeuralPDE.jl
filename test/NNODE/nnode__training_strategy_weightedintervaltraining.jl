using NeuralPDE
using Test

@testset "Training Strategy: WeightedIntervalTraining" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, Statistics

    Random.seed!(100)

    function f(u, p, t)
        [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
    end
    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0, 1.0]
    prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
    true_sol = solve(prob_oop, Tsit5(); saveat = 0.01)

    N = 64
    chain = Chain(
        Dense(1, N, gelu), Dense(N, N, gelu), Dense(N, N, gelu),
        Dense(N, N, gelu), Dense(N, length(u0))
    )

    alg = NNODE(
        chain, Adam(0.01); strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 200)
    )

    sol = solve(prob_oop, alg; verbose = false, maxiters = 5000, saveat = 0.01)
    @test abs(mean(sol) - mean(true_sol)) < 0.2
end
