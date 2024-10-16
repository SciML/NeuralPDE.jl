using Lux, Optimisers, OptimizationOptimisers, StochasticDiffEq, DiffEqNoiseProcess,
      OptimizationOptimJL, Test, Statistics, Random, NeuralPDE

Random.seed!(100)

@testset "RODE Example 1" begin
    linear = (u, p, t, W) -> 2u * sin(W)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    dt = 1 / 50.0f0
    W = WienerProcess(0.0, 0.0, nothing)
    prob = RODEProblem(linear, u0, tspan, noise = W)
    chain = Chain(Dense(2, 32, gelu), Dense(32, 32, gelu), Dense(32, 1))
    opt = Optimisers.Adam(1e-2)
    sol = solve(
        prob, NNRODE(chain, W, opt); dt, verbose = true, abstol = 1e-10, maxiters = 10000)
    W2 = NoiseWrapper(sol.W)
    prob1 = RODEProblem(linear, u0, tspan, noise = W2)
    sol2 = solve(prob1, RandomEM(), dt = dt)
    err = mean(abs2, sol.u .- sol2.u)
    @test err < 0.3
end

@testset "RODE Example 2" begin
    linear = (u, p, t, W) -> @.(t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
                                u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))+5 * W)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    dt = 1 / 100.0f0
    W = WienerProcess(0.0, 0.0, nothing)
    prob = RODEProblem(linear, u0, tspan, noise = W)
    chain = Chain(Dense(2, 32, gelu), Dense(32, 32, gelu), Dense(32, 1))
    opt = Optimisers.Adam(1e-2)
    sol = solve(
        prob, NNRODE(chain, W, opt); dt, verbose = true, abstol = 1e-10, maxiters = 10000)
    W2 = NoiseWrapper(sol.W)
    prob1 = RODEProblem(linear, u0, tspan, noise = W2)
    sol2 = solve(prob1, RandomEM(); dt)
    err = mean(abs2, sol.u .- sol2.u)
    @test err < 0.8
end
