using Test
using OrdinaryDiffEq, OptimizationOptimisers
using Lux
using Statistics, Random
using NeuralPDE

# dG(u(t,p),t) = u(t,p)
@testset "Example du = cos(p * t)" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    prob = ODEProblem(equation,  u0, tspan)

    branch = Lux.Chain(
        Lux.Dense(1, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast),
        Lux.Dense(10, 10))
    trunk = Lux.Chain(
        Lux.Dense(1, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast))

    deeponet = NeuralPDE.DeepONet(branch, trunk; linear = nothing)
    a = rand(1, 50, 40)
    b = rand(1, 1, 40)
    x = (branch = a, trunk = b)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet(x, θ, st)[1]

    bounds = (p = [0.1f0, pi],)

    strategy  = NeuralPDE.SomeStrategy(branch_size = 50, trunk_size = 40)

    opt = OptimizationOptimisers.Adam(0.03)
    alg = NeuralPDE.PINOODE(deeponet, opt, bounds; strategy = strategy)

    sol = solve(prob, alg, verbose = false, maxiters = 2000)

    ground_analytic = (u0, p, t) ->  u0 + sin(p * t) / (p)
    p_ = range(bounds.p[1], stop = bounds.p[2], length = strategy.branch_size)
    p = reshape(p_, 1, branch_size, 1)
    ground_solution = ground_analytic.(u0, p, sol.t.trunk)

    @test ground_solution≈sol.u rtol=0.01
end

@testset "Example du = p*t^2 " begin
    equation = (u, p, t) -> p * t^2
    tspan = (0.0f0, 1.0f0)
    u0 = 0.f0
    prob = ODEProblem(equation, u0, tspan)

    # init neural operator
    branch = Lux.Chain(
        Lux.Dense(1, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast))
    trunk = Lux.Chain(
        Lux.Dense(1, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast),
        Lux.Dense(10, 10, Lux.tanh_fast))
    linear = Lux.Chain(Lux.Dense(10, 1))
    deeponet = NeuralPDE.DeepONet(branch, trunk; linear= linear)

    a = rand(1, 50, 40)
    b = rand(1, 1, 40)
    x = (branch = a, trunk = b)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet(x, θ, st)[1]

    bounds = (p = [0.0f0, 2.f0],)
    strategy = NeuralPDE.SomeStrategy(branch_size = 50, trunk_size = 40)
    opt = OptimizationOptimisers.Adam(0.03)
    alg = NeuralPDE.PINOODE(deeponet, opt, bounds; strategy = strategy)

    sol = solve(prob, alg, verbose = false, maxiters = 2000)
    ground_analytic = (u0, p, t) -> u0 + p * t^3 / 3
    p_ = range(bounds.p[1], stop = bounds.p[2], length = strategy.branch_size)
    p = reshape(p_, 1, branch_size, 1)
    ground_solution = ground_analytic.(u0, p, sol.t.trunk)

    @test ground_solution≈sol.u rtol=0.01
end

# @testset "Example with data" begin
# end
