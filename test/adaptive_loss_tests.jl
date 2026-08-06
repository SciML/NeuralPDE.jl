@testsetup module AdaptiveLossTestSetup
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OptimizationOptimisers
import DomainSets: Interval

function solve_with_adaptive_loss(adaptive_loss; haslogger = false, outdir = nothing, run = 1)
    @parameters x, y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0,
        u(x, 1) ~ 0.0,
    ]

    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))

    strategy = GridTraining(0.05)

    discretization = PhysicsInformedNN(chain, strategy; adaptive_loss)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)

    res = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 500)
    phi = discretization.phi

    analytic_sol(x, y) = sin(pi * x) * sin(pi * y) / (2 * pi^2)

    xs = 0.0:0.05:1.0
    ys = 0.0:0.05:1.0
    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
    u_real = [analytic_sol(x, y) for x in xs for y in ys]
    return sum(abs2, u_predict .- u_real) / sum(abs2, u_real)
end
export solve_with_adaptive_loss
end

@testitem "NonAdaptiveLoss with Custom Weights" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    disc = PhysicsInformedNN(
        chain, GridTraining(0.25);
        adaptive_loss = NonAdaptiveLoss(; pde_loss_weights = 2.0, bc_loss_weights = 1.0)
    )
    prob = discretize(heat_sys, disc)
    @test prob isa Optimization.OptimizationProblem
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= 0
end

@testitem "MiniMaxAdaptiveLoss Reweighting" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    disc = PhysicsInformedNN(
        chain, GridTraining(0.25);
        adaptive_loss = MiniMaxAdaptiveLoss(10; pde_max_optimiser = OptimizationOptimisers.Adam(0.01),
                                              bc_max_optimiser = OptimizationOptimisers.Adam(0.01))
    )
    prob = discretize(heat_sys, disc)
    @test prob isa Optimization.OptimizationProblem

    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= 0

    sol = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 10)
    @test isfinite(prob.f(sol.u, nothing))
end

@testitem "additional_loss Function Support" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    custom_loss_fn(phi, θ, p) = 0.5 * sum(abs2, θ)

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
    discretization = PhysicsInformedNN(
        chain, GridTraining(0.2); additional_loss = custom_loss_fn
    )
    prob = discretize(heat_sys, discretization)

    @test prob isa Optimization.OptimizationProblem
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= custom_loss_fn(discretization.phi, prob.u0, nothing)
end

@testitem "2D Poisson NonAdaptiveLoss" tags = [:pinnparser] setup = [AdaptiveLossTestSetup] begin
    loss = NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)
    tmpdir = mktempdir()
    total_diff_rel = solve_with_adaptive_loss(loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
end

@testitem "2D Poisson MiniMaxAdaptiveLoss" tags = [:pinnparser] setup = [AdaptiveLossTestSetup] begin
    loss = MiniMaxAdaptiveLoss(100; pde_loss_weights = 1, bc_loss_weights = 1)
    tmpdir = mktempdir()
    total_diff_rel = solve_with_adaptive_loss(loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
end

@testitem "Approximation from data and additional_loss" tags = [:pinnparser] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Optimisers,
        ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Lux
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    @parameters x
    @variables u(..)

    eq = [u(0) ~ u(0)]
    bc = [u(0) ~ u(0)]
    x0 = 0
    x_end = pi
    dx = pi / 10

    domain = [x ∈ Interval(x0, x_end)]
    hidden = 10

    chain = Chain(
        Dense(1, hidden, tanh), Dense(hidden, hidden, sin),
        Dense(hidden, hidden, tanh), Dense(hidden, 1)
    )

    strategy = GridTraining(dx)
    xs = collect(x0:dx:x_end)'

    aproxf(x) = @. cospi(x)
    data = aproxf(xs)

    additional_loss(phi, θ, p) = sum(abs2, phi(xs, θ) .- data)

    discretization = PhysicsInformedNN(chain, strategy; additional_loss)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)

    res = solve(prob, Adam(0.01); maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 500)
    phi = discretization.phi

    @test phi(xs, res.u) ≈ aproxf(xs) rtol = 0.02
end