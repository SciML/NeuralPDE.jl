using OrdinaryDiffEq, Lux, OptimizationOptimisers, Test, Statistics, NeuralPDE

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
maxiters = 30000

prob_oop = ODEProblem{false}(fu, u0, tspan, p)
true_sol = solve(prob_oop, Tsit5(), saveat = saveat)
func = Lux.σ
N = 12
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
    Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))

opt = OptimizationOptimisers.Adam(0.01)
threshold = 0.2

#bad choices for weights, samples and dx so that the algorithm will fail without the added points
weights = [0.3, 0.3, 0.4]
points = 3
dx = 1.0

@testset "GridTraining" begin
    println("GridTraining")
    @testset "Without added points" begin
        println("Without added points")
        # (difference between solutions should be high)
        alg = NNODE(chain, opt, autodiff = false, strategy = GridTraining(dx))
        sol = solve(prob_oop, alg, verbose = false, maxiters = maxiters, saveat = saveat)
        @test abs(mean(sol) - mean(true_sol)) > threshold
    end
    @testset "With added points" begin
        println("With added points")
        # (difference between solutions should be low)
        alg = NNODE(chain, opt, autodiff = false, strategy = GridTraining(dx))
        sol = solve(prob_oop, alg, verbose = false, maxiters = maxiters,
            saveat = saveat, tstops = addedPoints)
        @test_broken abs(mean(sol) - mean(true_sol)) < threshold
    end
end

@testset "WeightedIntervalTraining" begin
    println("WeightedIntervalTraining")
    @testset "Without added points" begin
        println("Without added points")
        # (difference between solutions should be high)
        alg = NNODE(chain, opt, autodiff = false,
            strategy = WeightedIntervalTraining(weights, points))
        sol = solve(prob_oop, alg, verbose = false, maxiters = maxiters, saveat = saveat)
        @test abs(mean(sol) - mean(true_sol)) > threshold
    end
    @testset "With added points" begin
        println("With added points")
        # (difference between solutions should be low)
        alg = NNODE(chain, opt, autodiff = false,
            strategy = WeightedIntervalTraining(weights, points))
        sol = solve(prob_oop, alg, verbose = false, maxiters = maxiters,
            saveat = saveat, tstops = addedPoints)
        @test_broken abs(mean(sol) - mean(true_sol)) < threshold
    end
end

@testset "StochasticTraining" begin
    println("StochasticTraining")
    @testset "Without added points" begin
        println("Without added points")
        # (difference between solutions should be high)
        alg = NNODE(chain, opt, autodiff = false, strategy = StochasticTraining(points))
        sol = solve(prob_oop, alg, verbose = false, maxiters = maxiters, saveat = saveat)
        @test abs(mean(sol) - mean(true_sol)) > threshold
    end
    @testset "With added points" begin
        println("With added points")
        # (difference between solutions should be low)
        alg = NNODE(chain, opt, autodiff = false, strategy = StochasticTraining(points))
        sol = solve(prob_oop, alg, verbose = false, maxiters = maxiters,
            saveat = saveat, tstops = addedPoints)
        @test_broken abs(mean(sol) - mean(true_sol)) < threshold
    end
end
