using NeuralPDE, Test

using ModelingToolkit, Optimization, OptimizationOptimisers, Distributions, MethodOfLines,
      OrdinaryDiffEq, LinearAlgebra
import ModelingToolkit: Interval, infimum, supremum

@testset "Poisson's equation" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    # Initial and boundary conditions
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
        u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    strategy = QuasiRandomTraining(256, minibatch = 32)
    discretization = DeepGalerkin(2, 1, 20, 3, tanh, tanh, identity, strategy)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)

    callback = function (p, l)
        p.iter % 50 == 0 && println("$(p.iter) => $l")
        return false
    end

    res = Optimization.solve(
        prob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = Optimization.solve(
        prob, OptimizationOptimisers.Adam(0.001); callback, maxiters = 200)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

    u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
        (length(xs), length(ys)))
    u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
        (length(xs), length(ys)))

    @test u_real≈u_predict atol=0.01 norm=Base.Fix2(norm, Inf)
end

@testset "Black-Scholes PDE: European Call Option" begin
    K = 50.0
    T = 1.0
    r = 0.05
    σ = 0.25
    S = 130.0
    S_multiplier = 1.3

    @parameters x t
    @variables g(..)
    G(x) = max(x - K, 0.0)

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    eq = Dt(g(t, x)) + r * x * Dx(g(t, x)) + 0.5 * σ^2 * Dxx(g(t, x)) ~ r * g(t, x)

    bcs = [g(T, x) ~ G(x)] # terminal condition

    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, S * S_multiplier)]

    strategy = QuasiRandomTraining(128, minibatch = 32)
    discretization = DeepGalerkin(2, 1, 40, 3, tanh, tanh, identity, strategy)

    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [g(t, x)])
    prob = discretize(pde_system, discretization)

    callback = function (p, l)
        p.iter % 50 == 0 && println("$(p.iter) => $l")
        return false
    end

    res = Optimization.solve(prob, Adam(0.1); callback, maxiters = 100)
    prob = remake(prob, u0 = res.u)
    res = Optimization.solve(prob, Adam(0.01); callback, maxiters = 500)
    phi = discretization.phi

    function analytical_soln(t, x, K, σ, T)
        d₊ = (log(x / K) + (r + 0.5 * σ^2) * (T - t)) / (σ * sqrt(T - t))
        d₋ = d₊ - (σ * sqrt(T - t))
        return x * cdf(Normal(0, 1), d₊) .- K * exp(-r * (T - t)) * cdf(Normal(0, 1), d₋)
    end
    analytic_sol_func(t, x) = analytical_soln(t, x, K, σ, T)

    domains2 = [t ∈ Interval(0.0, T - 0.001), x ∈ Interval(0.0, S)]
    ts = collect(infimum(domains2[1].domain):0.01:supremum(domains2[1].domain))
    xs = collect(infimum(domains2[2].domain):1.0:supremum(domains2[2].domain))

    u_real = [analytic_sol_func(t, x) for t in ts, x in xs]
    u_predict = [first(phi([t, x], res.u)) for t in ts, x in xs]
    @test u_predict≈u_real rtol=0.05
end

@testset "Burger's equation" begin
    @parameters x t
    @variables u(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    α = 0.05
    eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - α * Dxx(u(t, x)) ~ 0 # Burger's equation

    bcs = [
        u(0.0, x) ~ -sin(π * x),
        u(t, -1.0) ~ 0.0,
        u(t, 1.0) ~ 0.0
    ]

    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(-1.0, 1.0)]

    # MethodOfLines
    dx = 0.01
    order = 2
    discretization = MOLFiniteDifference([x => dx], t, saveat = 0.01)
    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    prob = discretize(pde_system, discretization)
    sol = solve(prob, Tsit5())
    ts = sol[t]
    xs = sol[x]

    u_MOL = sol[u(t, x)]

    # NeuralPDE
    strategy = QuasiRandomTraining(256, minibatch = 32)
    discretization = DeepGalerkin(2, 1, 50, 5, tanh, tanh, identity, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    prob = discretize(pde_system, discretization)

    callback = function (p, l)
        p.iter % 50 == 0 && println("$(p.iter) => $l")
        return false
    end

    res = Optimization.solve(prob, Adam(0.01); callback = callback, maxiters = 200)
    prob = remake(prob, u0 = res.u)
    res = Optimization.solve(prob, Adam(0.001); callback = callback, maxiters = 100)
    phi = discretization.phi

    u_predict = [first(phi([t, x], res.u)) for t in ts, x in xs]

    @test u_predict≈u_MOL rtol=0.025
end
