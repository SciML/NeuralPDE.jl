using Test
using OrdinaryDiffEq, OptimizationOptimisers
using Lux
using Statistics, Random
using NeuralPDE

@testset "Example p" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    u0 = 0.0f0
    prob = ODEProblem(equation, u0, tspan)
    # prob = PINOODEProblem(equation, tspan)?

    # init neural operator
    branch = Lux.Chain(Lux.Dense(1, 32, Lux.σ), Lux.Dense(32, 32, Lux.σ), Lux.Dense(32, 1))
    trunk = Lux.Chain(Lux.Dense(1, 32, Lux.σ), Lux.Dense(32, 32, Lux.σ), Lux.Dense(32, 1))
    deeponet = NeuralPDE.DeepONet(branch, trunk)

    θ, st = Lux.setup(Random.default_rng(), deeponet)
    a = rand(1, 10)
    t = rand(1, 10)
    x = (branch = a, trunk = t)
    y, st = deeponet(x, θ, st)

    bounds = (p = [0.1, pi / 2],)
    #instances_size = 100 TODO remove dt -> instances_size
    opt = OptimizationOptimisers.Adam(0.1)
    alg = NeuralPDE.PINOODE(deeponet, opt, bounds)
    sol, phi  = solve(prob, alg, dt = 0.1, verbose = true, maxiters = 2000)

    instances_size = 100
    p = range(bounds.p[1], stop = bounds.p[2], length = instances_size)
    t = range(tspan[1], stop = tspan[2], length = instances_size)
    x = (branch = collect(p)', trunk = collect(t)')
    predict = phi(x, sol.u)

    ground_func = (u0, p, t) -> u0 + sin(p * t) / (p)
    ground_solution = ground_func.(u0, x.branch', x.trunk)
    @test ground_solution ≈ predict atol=1
end

@testset "Example with data" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    u0 = 0.0f0
    prob = ODEProblem(linear, u0, tspan)

    # init neural operator
    deeponet = DeepONet(branch, trunk)
    opt = OptimizationOptimisers.Adam(0.01)
    bounds = (p = [0, pi / 2])
    function data_loss()
        #code
    end
    alg = NeuralPDE.PINOODE(chain, opt, bounds; add_loss = data_loss)
    sol = solve(prob, alg, verbose = false, maxiters = 2000)
    predict = sol.predict

    ground_func = (u0, p, t) -> u0 + sin(p * t) / (p)
    ground = ground_func(..)
    @test ground≈predict atol=1
end

@testset "Example u0" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    p = Float32(pi)
    # u0 = 2.0f0
    prob = ODEProblem(linear, 0, tspan, p)
    prob = PINOODEProblem(linear, tspan, p)

    neuraloperator = DeepONet(branch, trunk)

    opt = OptimizationOptimisers.Adam(0.001)
    bounds = (u0 = [0, 2],)
    alg = PINOODE(chain, opt, bounds)
    pino_solution = solve(prob, alg, verbose = false, maxiters = 2000)
    predict = pino_solution.predict

    ground_func = (u0, p, t) -> u0 + sin(p * t) / (p)
    ground = ground_func(...)
    @test ground≈predict atol=1.0
end

@testset "Example u0 and p" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    p = Float32(pi)
    # u0 = 2.0f0
    prob = ODEProblem(linear, 0, tspan, p)
    prob = PINOODEProblem(linear, tspan, p)

    neuraloperator = DeepONet(branch, trunk)

    opt = OptimizationOptimisers.Adam(0.001)
    bounds = (u0 = [0, 2],)
    alg = PINOODE(chain, opt, bounds)
    pino_solution = solve(prob, alg, verbose = false, maxiters = 2000)
    predict = pino_solution.predict

    ground_func = (u0, p, t) -> u0 + sin(p * t) / (p)
    ground = ground_func(...)
    @test ground≈predict atol=1.0
end
