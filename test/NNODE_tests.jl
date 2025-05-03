@testitem "Scalar" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    using OptimizationOptimJL: BFGS

    Random.seed!(100)

    linear = (u, p, t) -> cospi(2t)
    tspan = (0.0f0, 1.0f0)
    u0 = 0.0f0
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
        autodiff in [false, true]

        if autodiff
            @test_throws ArgumentError solve(
                prob, NNODE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0)
            continue
        end

        @testset for (dt, abstol) in [(1 / 20.0f0, 1e-10), (nothing, 1e-6)]
            kwargs = (; verbose = false, dt, abstol, maxiters = 200)
            sol = solve(prob, NNODE(luxchain, opt; autodiff); kwargs...)
        end
    end
end

@testitem "Vector" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    using OptimizationOptimJL: BFGS

    Random.seed!(100)

    linear = (u, p, t) -> [cospi(2t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0]
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
        autodiff in [false, true]

        sol = solve(
            prob, NNODE(luxchain, opt); verbose = false, maxiters = 200, abstol = 1e-6)

        @test sol(0.5) isa Vector
        @test sol(0.5; idxs = 1) isa Number
        @test sol.k isa SciMLBase.OptimizationSolution
    end
end

@testitem "ODE I" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    linear = (u, p, t) -> @. t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
                             u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))
    linear_analytic = (u0, p, t) -> [exp(-(t^2) / 2) / (1 + t + t^3) + t^2]
    prob = ODEProblem(
        ODEFunction(linear; analytic = linear_analytic), [1.0f0], (0.0f0, 1.0f0))
    luxchain = Chain(Dense(1, 128, σ), Dense(128, 1))
    opt = Adam(0.01)

    @testset for strategy in [nothing, StochasticTraining(100)], batch in [false, true]
        sol = solve(
            prob, NNODE(luxchain, opt; batch, strategy); verbose = false, maxiters = 200,
            abstol = 1e-6)
        @test sol.errors[:l2] < 0.5
    end
end

@testitem "ODE Example 2" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

    linear = (u, p, t) -> -u / 5 + exp(-t / 5) .* cos(t)
    linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
    prob = ODEProblem(
        ODEFunction(linear; analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))
    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))

    @testset for batch in [false, true], strategy in [StochasticTraining(100), nothing]
        opt = Adam(0.1)
        sol = solve(
            prob, NNODE(luxchain, opt; batch, strategy); verbose = false, maxiters = 200,
            abstol = 1e-6)
        @test sol.errors[:l2] < 0.5
    end
end

@testitem "ODE Example 3" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

    linear = (u, p, t) -> [cospi(2t), sinpi(2t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0, -1.0f0 / 2pi]
    linear_analytic = (u0, p, t) -> [sinpi(2t) / 2pi, -cospi(2t) / 2pi]
    odefunction = ODEFunction(linear; analytic = linear_analytic)
    prob = ODEProblem(odefunction, u0, tspan)
    luxchain = Chain(Dense(1, 10, σ), Dense(10, 2))
    opt = Adam(0.1)
    alg = NNODE(luxchain, opt; autodiff = false)

    sol = solve(
        prob, alg; verbose = false, maxiters = 1000, abstol = 1e-6, saveat = 0.01)

    @test sol.errors[:l2] < 0.5
end

@testitem "Training Strategy: WeightedIntervalTraining" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, Statistics

    Random.seed!(100)

    function f(u, p, t)
        [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
    end
    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0, 1.0]
    prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
    true_sol = solve(prob_oop, Tsit5(); saveat = 0.01)

    N = 64
    chain = Chain(Dense(1, N, gelu), Dense(N, N, gelu), Dense(N, N, gelu),
        Dense(N, N, gelu), Dense(N, length(u0)))

    alg = NNODE(
        chain, Adam(0.01); strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 200))

    sol = solve(prob_oop, alg; verbose = false, maxiters = 5000, saveat = 0.01)
    @test abs(mean(sol) - mean(true_sol)) < 0.2
end

@testitem "Training Strategy: Others" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, Integrals

    Random.seed!(100)

    linear = (u, p, t) -> cospi(2t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sinpi(2t)
    tspan = (0.0, 1.0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(ODEFunction(linear; analytic = linear_analytic), 0.0, (0.0, 1.0))
    opt = Adam(0.1, (0.9, 0.95))
    u_analytical(x) = (1 / (2pi)) .* sinpi.(2x)

    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))
    u_, t_ = u_analytical(ts), ts

    function additional_loss(phi, θ)
        return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
    end

    @testset "$(nameof(typeof(strategy)))" for strategy in [
        GridTraining(0.01),
        StochasticTraining(1000),
        QuadratureTraining(reltol = 1e-3, abstol = 1e-6, maxiters = 50,
            batch = 100, quadrature_alg = QuadGKJL())
    ]
        alg = NNODE(luxchain, opt; additional_loss, strategy)
        @test begin
            sol = solve(prob, alg; verbose = false, maxiters = 500, abstol = 1e-6)
            sol.errors[:l2] < 0.5
        end
    end
end

@testitem "ODE Parameter Estimation" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, OptimizationOptimJL, LineSearches
    Random.seed!(100)

    function lorenz(u, p, t)
        return [p[1] * (u[2] - u[1]),
            u[1] * (p[2] - u[3]) - u[2],
            u[1] * u[2] - p[3] * u[3]]
    end
    tspan = (0.0, 1.0)
    prob = ODEProblem(lorenz, [1.0, 0.0, 0.0], tspan, [1.0, 1.0, 1.0])
    true_p = [2.0, 3.0, 2.0]
    prob2 = remake(prob, p = true_p)
    n = 8
    luxchain = Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, 3))
    sol = solve(prob2, Tsit5(); saveat = 0.01)
    t_ = sol.t
    u_ = sol.u
    sol_points = hcat(u_...)
    u1_ = [u_[i][1] for i in eachindex(t_)]
    u2_ = [u_[i][2] for i in eachindex(t_)]
    u3_ = [u_[i][3] for i in eachindex(t_)]
    dataset = [u1_, u2_, u3_, t_, ones(length(t_))]

    alg = NNODE(luxchain, BFGS(linesearch = BackTracking());
        strategy = GridTraining(0.01), dataset = dataset,
        param_estim = true)
    sol = solve(prob, alg; verbose = false, abstol = 1e-8, maxiters = 1000, saveat = t_)

    @test sol.k.u.p≈true_p atol=1e-2
    @test reduce(hcat, sol.u)≈sol_points atol=1e-2
end

@testitem "ODE Parameter Estimation Improvement" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, OptimizationOptimJL, LineSearches
    using FastGaussQuadrature, PolyChaos, Integrals
    Random.seed!(100)

    function lorenz(u, p, t)
        return [p[1] * (u[2] - u[1]),
            u[1] * (p[2] - u[3]) - u[2],
            u[1] * u[2] - p[3] * u[3]]
    end
    tspan = (0.0, 5.0)
    prob = ODEProblem(lorenz, [1.0, 0.0, 0.0], tspan, [-10.0, -10.0, -10.0])
    true_p = [2.0, 3.0, 2.0]
    prob2 = remake(prob, p = true_p)
    n = 8
    luxchain = Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, 3))

    # this example is especially easy for new loss.
    # even with 1 observed data points, we can exactly calculate the physics parameters (even before solve call).
    N = 7
    # x, w = gausslegendre(N) # does not include endpoints
    x, w = gausslobatto(N)
    # x, w = clenshaw_curtis(N)
    a = tspan[1]
    b = tspan[2]

    # transform the roots and weights
    # x = map((x) -> (2 * (t - a) / (b - a)) - 1, x)
    t = map((x) -> (x * (b - a) + (b + a)) / 2, x)
    W = map((x) -> x * (b - a) / 2, w)
    sol = solve(prob2, Tsit5(); saveat = t)
    t_ = sol.t
    u_ = sol.u
    u1_ = [u_[i][1] for i in eachindex(t_)]
    u2_ = [u_[i][2] for i in eachindex(t_)]
    u3_ = [u_[i][3] for i in eachindex(t_)]
    dataset = [u1_, u2_, u3_, t_, W]

    alg_old = NNODE(luxchain, BFGS(linesearch = BackTracking());
        strategy = GridTraining(0.01), dataset = dataset,
        param_estim = true)
    sol_old = solve(
        prob, alg_old; verbose = false, abstol = 1e-12, maxiters = 2000, saveat = 0.01)

    alg_new = NNODE(
        luxchain, BFGS(linesearch = BackTracking()); strategy = GridTraining(0.01),
        param_estim = true, dataset = dataset, estim_collocate = true)
    sol_new = solve(
        prob, alg_new; verbose = false, abstol = 1e-12, maxiters = 2000, saveat = 0.01)

    sol = solve(prob2, Tsit5(); saveat = 0.01)
    sol_points = hcat(sol.u...)
    sol_old_points = hcat(sol_old.u...)
    sol_new_points = hcat(sol_new.u...)

    @test !isapprox(sol_old.k.u.p, true_p; atol = 10)
    @test !isapprox(sol_old_points, sol_points; atol = 10)

    @test sol_new.k.u.p≈true_p atol=1e-2
    @test sol_new_points≈sol_points atol=3e-2
end

@testitem "ODE Complex Numbers" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

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

    chain = Chain(
        Dense(1, 16, tanh; init_weight = kaiming_normal(ComplexF64)),
        Dense(16, 4; init_weight = kaiming_normal(ComplexF64))
    )
    ps, st = Lux.setup(Random.default_rng(), chain)

    ground_truth = solve(problem, Tsit5(), saveat = 0.01)

    strategies = [
        StochasticTraining(500),
        GridTraining(0.01),
        WeightedIntervalTraining([0.1, 0.4, 0.4, 0.1], 500)
    ]

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        alg = NNODE(chain, Adam(0.01); strategy)
        sol = solve(problem, alg; verbose = false, maxiters = 5000, saveat = 0.01)
        @test sol.u≈ground_truth.u rtol=1e-1
    end

    alg = NNODE(chain, Adam(0.01); strategy = QuadratureTraining())
    @test_throws ErrorException solve(
        problem, alg; verbose = false, maxiters = 5000, saveat = 0.01)
end

@testitem "NNODE: Translating from Flux" tags=[:nnode] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    import Flux

    Random.seed!(100)

    linear = (u, p, t) -> cospi(2t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sinpi(2t)
    tspan = (0.0f0, 1.0f0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(
        ODEFunction(linear; analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))

    u_analytical(x) = (1 / (2pi)) .* sinpi.(2x)
    fluxchain = Flux.Chain(Flux.Dense(1, 5, Flux.σ), Flux.Dense(5, 1))
    opt = Adam(0.1)

    alg = NNODE(fluxchain, opt)
    @test alg.chain isa Lux.AbstractLuxLayer
    sol = solve(prob, alg; verbose = false, abstol = 1e-10, maxiters = 200)
    @test sol.errors[:l2] < 0.5
end

@testitem "Training Strategy with `tstops`" tags=[:nnode] begin
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
    chain = Chain(Dense(1 => N, σ), Dense(N => N, σ), Dense(N => N, σ), Dense(N => N, σ),
        Dense(N => length(u0)))

    threshold = 0.2

    @testset "$(nameof(typeof(strategy)))" for strategy in [
        GridTraining(1.0),
        WeightedIntervalTraining([0.3, 0.3, 0.4], 3),
        StochasticTraining(3)
    ]
        alg = NNODE(chain, Adam(0.01); strategy, tstops = addedPoints)

        @testset "Without added points" begin
            sol = solve(prob_oop, alg; verbose = false, maxiters = 10000, saveat)
            @test abs(mean(sol) - mean(true_sol)) ≥ threshold
        end

        @testset "With added points" begin
            sol = solve(prob_oop, alg; verbose = false,
                maxiters = 10000, saveat, tstops = addedPoints)
            @test abs(mean(sol) - mean(true_sol)) < threshold
        end
    end
end
