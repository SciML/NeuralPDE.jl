using Test
using Random, NeuralPDE
using OrdinaryDiffEq, Statistics
import Lux, OptimizationOptimisers, OptimizationOptimJL
using Flux

Random.seed!(100)

@testset "Scalar" begin
    # Run a solve on scalars
    linear = (u, p, t) -> cos(2pi * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 0.0f0
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
    opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

    sol = solve(prob, NNODE(luxchain, opt), dt = 1 / 20.0f0, verbose = true,
                abstol = 1.0f-10, maxiters = 200)

    @test_throws ArgumentError solve(prob, NNODE(luxchain, opt; autodiff = true),
                        dt = 1 / 20.0f0,
                        verbose = true, abstol = 1.0f-10, maxiters = 200)

    sol = solve(prob, NNODE(luxchain, opt), verbose = true,
                abstol = 1.0f-6, maxiters = 200)

    opt = OptimizationOptimJL.BFGS()
    sol = solve(prob, NNODE(luxchain, opt), dt = 1 / 20.0f0, verbose = true,
                abstol = 1.0f-10, maxiters = 200)

    sol = solve(prob, NNODE(luxchain, opt), verbose = true,
                abstol = 1.0f-6, maxiters = 200)
end

@testset "Vector" begin
    # Run a solve on vectors
    linear = (u, p, t) -> [cos(2pi * t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0]
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

    opt = OptimizationOptimJL.BFGS()
    sol = solve(prob, NNODE(luxchain, opt), dt = 1 / 20.0f0, abstol = 1e-10,
                verbose = true, maxiters = 200)

    @test_throws ArgumentError solve(prob, NNODE(luxchain, opt; autodiff = true),
                        dt = 1 / 20.0f0,
                        abstol = 1e-10, verbose = true, maxiters = 200)

    sol = solve(prob, NNODE(luxchain, opt), abstol = 1.0f-6,
                verbose = true, maxiters = 200)

    @test sol(0.5) isa Vector
    @test sol(0.5; idxs = 1) isa Number
    @test sol.k isa SciMLBase.OptimizationSolution
end

@testset "Example 1" begin
    linear = (u, p, t) -> @. t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
                            u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))
    linear_analytic = (u0, p, t) -> [exp(-(t^2) / 2) / (1 + t + t^3) + t^2]
    prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), [1.0f0], (0.0f0, 1.0f0))
    luxchain = Lux.Chain(Lux.Dense(1, 128, Lux.σ), Lux.Dense(128, 1))
    opt = OptimizationOptimisers.Adam(0.01)

    sol = solve(prob, NNODE(luxchain, opt), verbose = true, maxiters = 400)
    @test sol.errors[:l2] < 0.5

    @test_throws AssertionError solve(prob, NNODE(luxchain, opt; batch = true), verbose = true,
                        maxiters = 400)

    sol = solve(prob,
                NNODE(luxchain, opt; batch = false,
                                strategy = StochasticTraining(100)),
                verbose = true, maxiters = 400)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob,
                NNODE(luxchain, opt; batch = true,
                                strategy = StochasticTraining(100)),
                verbose = true, maxiters = 400)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = false), verbose = true,
                maxiters = 400, dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = true), verbose = true,
                maxiters = 400,
                dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5
end

@testset "Example 2" begin
    linear = (u, p, t) -> -u / 5 + exp(-t / 5) .* cos(t)
    linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
    prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

    opt = OptimizationOptimisers.Adam(0.1)
    sol = solve(prob, NNODE(luxchain, opt), verbose = true, maxiters = 400,
                abstol = 1.0f-8)
    @test sol.errors[:l2] < 0.5

    @test_throws AssertionError solve(prob, NNODE(luxchain, opt; batch = true), verbose = true,
                        maxiters = 400,
                        abstol = 1.0f-8)

    sol = solve(prob,
                NNODE(luxchain, opt; batch = false,
                                strategy = StochasticTraining(100)),
                verbose = true, maxiters = 400,
                abstol = 1.0f-8)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob,
                NNODE(luxchain, opt; batch = true,
                                strategy = StochasticTraining(100)),
                verbose = true, maxiters = 400,
                abstol = 1.0f-8)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = false), verbose = true,
                maxiters = 400,
                abstol = 1.0f-8, dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = true), verbose = true,
                maxiters = 400,
                abstol = 1.0f-8, dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5
end

@testset "Example 3" begin
    linear = (u, p, t) -> [cos(2pi * t), sin(2pi * t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0, -1.0f0 / 2pi]
    linear_analytic = (u0, p, t) -> [sin(2pi * t) / 2pi, -cos(2pi * t) / 2pi]
    odefunction = ODEFunction(linear, analytic = linear_analytic)
    prob = ODEProblem(odefunction, u0, tspan)
    luxchain = Lux.Chain(Lux.Dense(1, 10, Lux.σ), Lux.Dense(10, 2))
    opt = OptimizationOptimisers.Adam(0.1)
    alg = NNODE(luxchain, opt; autodiff = false)

    sol = solve(prob,
                alg, verbose = true, dt = 1 / 40.0f0,
                maxiters = 2000, abstol = 1.0f-7)
    @test sol.errors[:l2] < 0.5
end

@testset "Training Strategies" begin
    @testset "WeightedIntervalTraining" begin
        function f(u, p, t)
            [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
        end
        p = [1.5, 1.0, 3.0, 1.0]
        u0 = [1.0, 1.0]
        prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
        true_sol = solve(prob_oop, Tsit5(), saveat = 0.01)
        func = Lux.σ
        N = 12
        chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                        Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))
        opt = OptimizationOptimisers.Adam(0.01)
        weights = [0.7, 0.2, 0.1]
        points = 200
        alg = NNODE(chain, opt, autodiff = false,
                            strategy = NeuralPDE.WeightedIntervalTraining(weights, points))
        sol = solve(prob_oop, alg, verbose = true, maxiters = 100000, saveat = 0.01)
        @test abs(mean(sol) - mean(true_sol)) < 0.2
    end

    linear = (u, p, t) -> cos(2pi * t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sin(2pi * t)
    tspan = (0.0, 1.0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0, (0.0, 1.0))
    opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))
    u_analytical(x) = (1 / (2pi)) .* sin.(2pi .* x)

    @testset "GridTraining" begin
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, strategy = GridTraining(0.01),
                            additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = true, abstol = 1e-8, maxiters = 500)
        @test sol1.errors[:l2] < 0.5
    end

    @testset "QuadratureTraining" begin
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = true, abstol = 1e-10, maxiters = 200)
        @test sol1.errors[:l2] < 0.5
    end

    @testset "StochasticTraining" begin
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, strategy = StochasticTraining(1000),
                            additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = true, abstol = 1e-8, maxiters = 500)
        @test sol1.errors[:l2] < 0.5
    end
end

@testset "Translating from Flux" begin
    linear = (u, p, t) -> cos(2pi * t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sin(2pi * t)
    tspan = (0.0, 1.0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0, (0.0, 1.0))
    opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))
    u_analytical(x) = (1 / (2pi)) .* sin.(2pi .* x)
    fluxchain = Flux.Chain(Flux.Dense(1, 5, Flux.σ), Flux.Dense(5, 1))
    alg1 = NNODE(fluxchain, opt)
    @test alg1.chain isa Lux.AbstractExplicitLayer
    sol1 = solve(prob, alg1, verbose = true, abstol = 1e-10, maxiters = 200)
    @test sol1.errors[:l2] < 0.5
end
