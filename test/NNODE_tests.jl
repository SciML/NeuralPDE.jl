using Test
using Random, NeuralPDE
using OrdinaryDiffEq, Statistics
import Lux, OptimizationOptimisers, OptimizationOptimJL
using Flux
using LineSearches

rng = Random.default_rng()
Random.seed!(100)

@testset "Scalar" begin
    # Run a solve on scalars
    println("Scalar")
    linear = (u, p, t) -> cos(2pi * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 0.0f0
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
    opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

    sol = solve(prob, NNODE(luxchain, opt), dt = 1 / 20.0f0, verbose = false,
        abstol = 1.0f-10, maxiters = 200)

    @test_throws ArgumentError solve(prob, NNODE(luxchain, opt; autodiff = true),
        dt = 1 / 20.0f0,
        verbose = false, abstol = 1.0f-10, maxiters = 200)

    sol = solve(prob, NNODE(luxchain, opt), verbose = false,
        abstol = 1.0f-6, maxiters = 200)

    opt = OptimizationOptimJL.BFGS()
    sol = solve(prob, NNODE(luxchain, opt), dt = 1 / 20.0f0, verbose = false,
        abstol = 1.0f-10, maxiters = 200)

    sol = solve(prob, NNODE(luxchain, opt), verbose = false,
        abstol = 1.0f-6, maxiters = 200)
end

@testset "Vector" begin
    # Run a solve on vectors
    println("Vector")
    linear = (u, p, t) -> [cos(2pi * t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0]
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

    opt = OptimizationOptimJL.BFGS()
    sol = solve(prob, NNODE(luxchain, opt), dt = 1 / 20.0f0, abstol = 1e-10,
        verbose = false, maxiters = 200)

    @test_throws ArgumentError solve(prob, NNODE(luxchain, opt; autodiff = true),
        dt = 1 / 20.0f0,
        abstol = 1e-10, verbose = false, maxiters = 200)

    sol = solve(prob, NNODE(luxchain, opt), abstol = 1.0f-6,
        verbose = false, maxiters = 200)

    @test sol(0.5) isa Vector
    @test sol(0.5; idxs = 1) isa Number
    @test sol.k isa SciMLBase.OptimizationSolution
end

@testset "Example 1" begin
    println("Example 1")
    linear = (u, p, t) -> @. t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
                             u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))
    linear_analytic = (u0, p, t) -> [exp(-(t^2) / 2) / (1 + t + t^3) + t^2]
    prob = ODEProblem(
        ODEFunction(linear, analytic = linear_analytic), [1.0f0], (0.0f0, 1.0f0))
    luxchain = Lux.Chain(Lux.Dense(1, 128, Lux.σ), Lux.Dense(128, 1))
    opt = OptimizationOptimisers.Adam(0.01)

    sol = solve(prob, NNODE(luxchain, opt), verbose = false, maxiters = 400)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob,
        NNODE(luxchain, opt; batch = false,
            strategy = StochasticTraining(100)),
        verbose = false, maxiters = 400)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob,
        NNODE(luxchain, opt; batch = true,
            strategy = StochasticTraining(100)),
        verbose = false, maxiters = 400)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = false), verbose = false,
        maxiters = 400, dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = true), verbose = false,
        maxiters = 400,
        dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5
end

@testset "Example 2" begin
    println("Example 2")
    linear = (u, p, t) -> -u / 5 + exp(-t / 5) .* cos(t)
    linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
    prob = ODEProblem(
        ODEFunction(linear, analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

    opt = OptimizationOptimisers.Adam(0.1)
    sol = solve(prob, NNODE(luxchain, opt), verbose = false, maxiters = 400,
        abstol = 1.0f-8)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob,
        NNODE(luxchain, opt; batch = false,
            strategy = StochasticTraining(100)),
        verbose = false, maxiters = 400,
        abstol = 1.0f-8)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob,
        NNODE(luxchain, opt; batch = true,
            strategy = StochasticTraining(100)),
        verbose = false, maxiters = 400,
        abstol = 1.0f-8)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = false), verbose = false,
        maxiters = 400,
        abstol = 1.0f-8, dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5

    sol = solve(prob, NNODE(luxchain, opt; batch = true), verbose = false,
        maxiters = 400,
        abstol = 1.0f-8, dt = 1 / 5.0f0)
    @test sol.errors[:l2] < 0.5
end

@testset "Example 3" begin
    println("Example 3")
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
        alg, verbose = false, dt = 1 / 40.0f0,
        maxiters = 2000, abstol = 1.0f-7)
    @test sol.errors[:l2] < 0.5
end

@testset "Training Strategies" begin
    @testset "WeightedIntervalTraining" begin
        println("WeightedIntervalTraining")
        function f(u, p, t)
            [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
        end
        p = [1.5, 1.0, 3.0, 1.0]
        u0 = [1.0, 1.0]
        prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
        true_sol = solve(prob_oop, Tsit5(), saveat = 0.01)
        func = Lux.σ
        N = 12
        chain = Lux.Chain(
            Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
            Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))
        opt = OptimizationOptimisers.Adam(0.01)
        weights = [0.7, 0.2, 0.1]
        points = 200
        alg = NNODE(chain, opt, autodiff = false,
            strategy = NeuralPDE.WeightedIntervalTraining(weights, points))
        sol = solve(prob_oop, alg, verbose = false, maxiters = 5000, saveat = 0.01)
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
        println("GridTraining")
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, strategy = GridTraining(0.01),
            additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = false, abstol = 1e-8, maxiters = 500)
        @test sol1.errors[:l2] < 0.5
    end

    @testset "QuadratureTraining" begin
        println("QuadratureTraining")
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = false, abstol = 1e-10, maxiters = 200)
        @test sol1.errors[:l2] < 0.5
    end

    @testset "StochasticTraining" begin
        println("StochasticTraining")
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, strategy = StochasticTraining(1000),
            additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = false, abstol = 1e-8, maxiters = 500)
        @test sol1.errors[:l2] < 0.5
    end
end

@testset "Parameter Estimation" begin
    println("Parameter Estimation")
    function lorenz(u, p, t)
        return [p[1] * (u[2] - u[1]),
            u[1] * (p[2] - u[3]) - u[2],
            u[1] * u[2] - p[3] * u[3]]
    end
    prob = ODEProblem(lorenz, [1.0, 0.0, 0.0], (0.0, 1.0), [1.0, 1.0, 1.0])
    true_p = [2.0, 3.0, 2.0]
    prob2 = remake(prob, p = true_p)
    sol = solve(prob2, Tsit5(), saveat = 0.01)
    t_ = sol.t
    u_ = reduce(hcat, sol.u)
    function additional_loss(phi, θ)
        return sum(abs2, phi(t_, θ) .- u_) / 100
    end
    n = 8
    luxchain = Lux.Chain(
        Lux.Dense(1, n, Lux.σ),
        Lux.Dense(n, n, Lux.σ),
        Lux.Dense(n, n, Lux.σ),
        Lux.Dense(n, 3)
    )
    opt = OptimizationOptimJL.BFGS(linesearch = BackTracking())
    alg = NNODE(luxchain, opt, strategy = GridTraining(0.01),
        param_estim = true, additional_loss = additional_loss)
    sol = solve(prob, alg, verbose = false, abstol = 1e-8, maxiters = 1000, saveat = t_)
    @test sol.k.u.p≈true_p atol=1e-2
    @test reduce(hcat, sol.u)≈u_ atol=1e-2
end

@testset "Complex Numbers" begin
    function bloch_equations(u, p, t)
        Ω, Δ, Γ = p
        γ = Γ / 2
        ρ₁₁, ρ₂₂, ρ₁₂, ρ₂₁ = u
        d̢ρ = [im * Ω * (ρ₁₂ - ρ₂₁) + Γ * ρ₂₂;
               -im * Ω * (ρ₁₂ - ρ₂₁) - Γ * ρ₂₂;
               -(γ + im * Δ) * ρ₁₂ - im * Ω * (ρ₂₂ - ρ₁₁);
               conj(-(γ + im * Δ) * ρ₁₂ - im * Ω * (ρ₂₂ - ρ₁₁))]
        return d̢ρ
    end

    u0 = zeros(ComplexF64, 4)
    u0[1] = 1
    time_span = (0.0, 2.0)
    parameters = [2.0, 0.0, 1.0]

    problem = ODEProblem(bloch_equations, u0, time_span, parameters)

    chain = Lux.Chain(
        Lux.Dense(1, 16, tanh;
            init_weight = (rng, a...) -> Lux.kaiming_normal(rng, ComplexF64, a...)),
        Lux.Dense(
            16, 4; init_weight = (rng, a...) -> Lux.kaiming_normal(rng, ComplexF64, a...))
    )
    ps, st = Lux.setup(rng, chain)

    opt = OptimizationOptimisers.Adam(0.01)
    ground_truth = solve(problem, Tsit5(), saveat = 0.01)
    strategies = [StochasticTraining(500), GridTraining(0.01),
        WeightedIntervalTraining([0.1, 0.4, 0.4, 0.1], 500)]

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        alg = NNODE(chain, opt, ps; strategy)
        sol = solve(problem, alg, verbose = false, maxiters = 5000, saveat = 0.01)
        @test sol.u≈ground_truth.u rtol=1e-1
    end

    alg = NNODE(chain, opt, ps; strategy = QuadratureTraining())
    @test_throws ErrorException solve(
        problem, alg, verbose = false, maxiters = 5000, saveat = 0.01)
end

@testset "Translating from Flux" begin
    println("Translating from Flux")
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
    sol1 = solve(prob, alg1, verbose = false, abstol = 1e-10, maxiters = 200)
    @test sol1.errors[:l2] < 0.5
end
