using NeuralPDE
using Test

@testset "Training Strategy with `tstops`" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, Statistics

    Random.seed!(100)

    function f(u, p, t)
        [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
    end
    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0, 1.0]

    tspan = (0.0, 3.0)
    points1 = rand(280)
    points2 = rand(80) .+ 1
    points3 = rand(40) .+ 2
    addedPoints = vcat(points1, points2, points3)

    saveat = 0.01

    prob_oop = ODEProblem{false}(f, u0, tspan, p)
    true_sol = solve(prob_oop, Tsit5(); saveat)
    N = 16
    chain = Chain(
        Dense(1 => N, σ), Dense(N => N, σ), Dense(N => N, σ), Dense(N => N, σ),
        Dense(N => length(u0))
    )

    threshold = 0.2

    @testset "$(nameof(typeof(strategy)))" for strategy in [
            GridTraining(1.0),
            WeightedIntervalTraining([0.3, 0.3, 0.4], 3),
            StochasticTraining(3),
        ]
        alg = NNODE(chain, Adam(0.01); strategy, tstops = addedPoints)

        @testset "Without added points" begin
            sol = solve(prob_oop, alg; verbose = false, maxiters = 10000, saveat)
            @test abs(mean(sol) - mean(true_sol)) ≥ threshold
        end

        @testset "With added points" begin
            sol = solve(
                prob_oop, alg; verbose = false,
                maxiters = 10000, saveat, tstops = addedPoints
            )
            @test abs(mean(sol) - mean(true_sol)) < threshold
        end
    end
end
