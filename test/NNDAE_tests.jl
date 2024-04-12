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
    tspan = (0.0f0, 1.0f0)

    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

    example = (du, u, p, t) -> [cos(2pi * t) - du[1], u[2] + cos(2pi * t) - du[2]]
    differential_vars = [true, false]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, cos), Lux.Dense(15, 15, sin), Lux.Dense(15, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    alg = NeuralPDE.NNDAE(chain, opt; autodiff = false)

    sol = solve(prob,
        alg, verbose = false, dt = 1 / 100.0f0,
        maxiters = 3000, abstol = 1.0f-10)
    @test ground_sol(0:(1 / 100):1)≈sol atol=0.4
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
    tspan = (0.0f0, pi / 2.0f0)
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
        alg, verbose = false, dt = 1 / 100.0f0,
        maxiters = 3000, abstol = 1.0f-10)

    @test ground_sol(0:(1 / 100):(pi / 2))≈sol atol=0.4
end

@testset "WeightedIntervalTraining" begin
    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    u₀ = [1.0, -1.0]
    du₀ = [0.0, 0.0]
    p = 0
    tspan = (0.0f0, pi / 2.0f0)
    prob = DAEProblem(example, du₀, u₀, tspan, p; differential_vars = differential_vars) #Lack of parameter here??
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    weights = [0.7, 0.2, 0.1]
    points = 200
    alg = NNDAE(chain, opt,init_params = nothing;  autodiff = false,strategy = NeuralPDE.WeightedIntervalTraining(weights, points))
    sol = solve(prob, alg, verbose = false, maxiters = 5000, saveat = 0.01)
    @test ground_sol(0:(1 / 100):(pi / 2))≈sol atol=0.4
end


@testset "StochasticTraining" begin
    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    u₀ = [1.0, -1.0]
    du₀ = [0.0, 0.0]
    tspan = (0.0f0, pi / 2.0f0)
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    weights = [0.7, 0.2, 0.1]
    points = 200
    alg = NNDAE(chain, opt,init_params = nothing;  autodiff = false,strategy = StochasticTraining(100))
    sol = solve(prob, alg, verbose = false, maxiters = 5000, saveat = 0.01)
    @test ground_sol(0:(1 / 100):(pi / 2))≈sol atol=0.4
end


#=

@testset "WeightedIntervalTraining" begin
    println("WeightedIntervalTraining")
    function example3(du, u, p, t)
        du[1] =  -u[2]
        u[1] =  -u[2]
        return out
    end
    p = []
    u0 = [1.0/4.0, 1.0/4.0]
    du0 = [0,0]
    tspan = (0.0, 100000.0)
    differential_vars = [true, false]
    prob = DAEProblem(example3, du0, u0, tspan, differential_vars = differential_vars)
    true_out_1(t) = exp(-t)/4.0
    true_out_2(t) = -1.0 * exp(-t)/4.0
    func = Lux.σ
    N = 12
    chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))
    opt = OptimizationOptimisers.Adam(0.01)
    weights = [0.7, 0.2, 0.1]
    points = 200
    alg = NNDAE(chain, opt,init_params = nothing;  autodiff = false,strategy = NeuralPDE.WeightedIntervalTraining(weights, points))
    sol = solve(prob, alg, verbose = false, maxiters = 5000, saveat = 0.01)
    #@test abs(mean(sol) - mean(true_sol)) < 0.2
    """Sol would have 2 outputs: one for u[1] and the other for u[2] so just I need compute the total error for all t in tspan """
    total_error = 0
    for i in tspan
        total_error  = total_error + abs(sol(i) - [true_out_1(i) true_out_2(i)])
    end
    if total_error < 0.01
        print("It works!")
    else
        print("Total error exceeds bound")
    end
end

=#



