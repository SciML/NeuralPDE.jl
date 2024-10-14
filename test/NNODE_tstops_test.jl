using OrdinaryDiffEq, Lux, OptimizationOptimisers, Optimisers, Test, Statistics, NeuralPDE

function fu(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
tspan = (0.0, 3.0)
points1 = [rand() for i in 1:280]
points2 = [rand() + 1 for i in 1:80]
points3 = [rand() + 2 for i in 1:40]
addedPoints = vcat(points1, points2, points3)

saveat = 0.01

prob_oop = ODEProblem{false}(fu, u0, tspan, p)
true_sol = solve(prob_oop, Tsit5(); saveat)
N = 16
chain = Chain(
    Dense(1, N, σ), Dense(N, N, σ), Dense(N, N, σ), Dense(N, N, σ), Dense(N, length(u0)))

opt = Adam(0.01)
threshold = 0.2

@testset "$(nameof(typeof(strategy)))" for strategy in [
    GridTraining(1.0),
    WeightedIntervalTraining([0.3, 0.3, 0.4], 3),
    StochasticTraining(3)
]
    alg = NNODE(chain, opt; autodiff = false, strategy)

    @testset "Without added points" begin
        sol = solve(prob_oop, alg; verbose = false, maxiters = 1000, saveat)
        @test abs(mean(sol) - mean(true_sol)) > threshold
    end

    @testset "With added points" begin
        sol = solve(
            prob_oop, alg; verbose = false, maxiters = 10000, saveat, tstops = addedPoints)
        @test abs(mean(sol) - mean(true_sol)) < threshold
    end
end
