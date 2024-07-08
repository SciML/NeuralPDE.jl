using Test, Flux
using Random, NeuralPDE
using OrdinaryDiffEq, Statistics
import Lux, OptimizationOptimisers, OptimizationOptimJL

Random.seed!(100)

@testset "Example 1" begin
    function example1(du, u, p, t)
        du[1] = cos(2pi * t)
        du[2] = u[2] + cos(2pi * t)
        nothing
    end
    u₀ = [1.0, -1.0]
    du₀ = [0.0, 0.0]
    M = [1.0 0
         0 0]
    f = ODEFunction(example1, mass_matrix = M)
    tspan = (0.0, 1.0)

    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

    example = (du, u, p, t) -> [cos(2pi * t) - du[1], u[2] + cos(2pi * t) - du[2]]
    differential_vars = [true, false]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, cos), Lux.Dense(15, 15, sin), Lux.Dense(15, 2))
    opt = OptimizationOptimJL.BFGS(linesearch = BackTracking())
    alg = NNDAE(chain, opt; autodiff = false)

    sol = solve(prob,
        alg, verbose = false, dt = 1 / 100.0,
        maxiters = 3000, abstol = 1e-10)
    @test reduce(hcat, ground_sol(0:(1 / 100):1).u)≈reduce(hcat, sol.u) rtol=1e-1
end

@testset "Example 2" begin
    function example2(du, u, p, t)
        du[1] = u[1] - t
        du[2] = u[2] - t
        nothing
    end
    M = [0.0 0
         0 1]
    u₀ = [0.0, 0.0]
    du₀ = [0.0, 0.0]
    tspan = (0.0, pi / 2.0)
    f = ODEFunction(example2, mass_matrix = M)
    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    alg = NNDAE(chain, OptimizationOptimisers.Adam(0.1); autodiff = false)

    sol = solve(prob,
        alg, verbose = false, dt = 1 / 100.0,
        maxiters = 3000, abstol = 1e-10)

    @test reduce(hcat, ground_sol(0:(1 / 100):(pi / 2.0)).u)≈reduce(hcat, sol.u) rtol=1e-2
end

@testset "WeightedIntervalTraining" begin
    function example2(du, u, p, t)
        du[1] = u[1] - t
        du[2] = u[2] - t
        nothing
    end
    M = [0.0 0.0
         0.0 1.0]
    u₀ = [0.0, 0.0]
    du₀ = [0.0, 0.0]
    tspan = (0.0, pi / 2.0)
    f = ODEFunction(example2, mass_matrix = M)
    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    weights = [0.7, 0.2, 0.1]
    points = 200
    alg = NNDAE(chain, OptimizationOptimisers.Adam(0.1),
        strategy = WeightedIntervalTraining(weights, points); autodiff = false)

    sol = solve(prob,
        alg, verbose = false, dt = 1 / 100.0,
        maxiters = 3000, abstol = 1e-10)

    @test reduce(hcat, ground_sol(0:(1 / 100):(pi / 2.0)).u)≈reduce(hcat, sol.u) rtol=1e-2
end

@testset "StochasticTraining" begin
    function example2(du, u, p, t)
        du[1] = u[1] - t
        du[2] = u[2] - t
        nothing
    end
    M = [0.0 0.0
         0.0 1.0]
    u₀ = [0.0, 0.0]
    du₀ = [0.0, 0.0]
    tspan = (0.0, pi / 2.0)
    f = ODEFunction(example2, mass_matrix = M)
    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    alg = NeuralPDE.NNDAE(chain, OptimizationOptimisers.Adam(0.1),
        strategy = NeuralPDE.StochasticTraining(1000); autodiff = false)
    sol = solve(prob,
        alg, verbose = false, dt = 1 / 100.0,
        maxiters = 3000, abstol = 1e-10)
    @test reduce(hcat, ground_sol(0:(1 / 100):(pi / 2.0)).u)≈reduce(hcat, sol.u) rtol=1e-2
end

@testset "QuadratureTraining" begin
    function example2(du, u, p, t)
        du[1] = u[1] - t
        du[2] = u[2] - t
        nothing
    end
    M = [0.0 0.0
         0.0 1.0]
    u₀ = [0.0, 0.0]
    du₀ = [0.0, 0.0]
    tspan = (0.0, pi / 2.0)
    f = ODEFunction(example2, mass_matrix = M)
    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 2))
    opt = OptimizationOptimJL.BFGS(linesearch = BackTracking())
    alg = NeuralPDE.NNDAE(chain, opt; autodiff = false)
    sol = solve(prob, alg, verbose = true, maxiters = 6000, abstol = 1e-10, dt = 1/100.0)
    @test reduce(hcat, ground_sol(0:(1 / 100):(pi / 2.0)).u)≈reduce(hcat, sol.u) rtol=1e-2
end