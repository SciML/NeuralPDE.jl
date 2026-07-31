@testsetup module IDETestSetup
using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimJL, Symbolics
import DomainSets: Interval, ClosedInterval, UnitSquare

function callback(p, l)
    if p.iter == 1 || p.iter % 10 == 0
        println("Current loss is: $l after $(p.iter) iterations")
    end
    return false
end
export callback
end

@testitem "Integral Tests (Constant, Variable, 2D)" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Zygote, Test
    import DomainSets: Interval, ClosedInterval, UnitSquare

    @testset "1D IDE with Constant Bounds" begin
        @parameters t
        @variables i(..)
        Di = Differential(t)
        Ii = Integral(t in ClosedInterval(0, 2))
        eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(t)) ~ 1
        bcs = [i(0.0) ~ 0.0]
        domains = [t ∈ Interval(0.0, 2.0)]

        chain = Chain(Dense(1, 8, σ), Dense(8, 1))

        @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])

        symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
            pde_system, chain; n_interior = 4, n_bc = 4
        )
        theta0 = symbolic_loss.theta0
        @test isfinite(symbolic_loss.loss(theta0))
        grad = Zygote.gradient(symbolic_loss.loss, theta0)
        @test grad !== nothing
        @test all(isfinite, grad[1])

        discretization_sym = PhysicsInformedNN(
            chain, GridTraining(0.25)
        )
        prob_sym = discretize(pde_system, discretization_sym)

        @test prob_sym isa Optimization.OptimizationProblem

        loss_sym = prob_sym.f(prob_sym.u0, nothing)
        @test isfinite(loss_sym)

        grad_sym = Zygote.gradient(θ -> prob_sym.f(θ, nothing), prob_sym.u0)[1]
        @test grad_sym !== nothing
        @test all(isfinite, grad_sym)

        initial_loss = prob_sym.f(prob_sym.u0, nothing)
        sol = solve(prob_sym, OptimizationOptimisers.Adam(0.01); maxiters = 300)
        final_loss = prob_sym.f(sol.u, nothing)
        @test final_loss < initial_loss
    end

    @testset "1D IDE with Variable Upper Bound" begin
        @parameters t
        @variables i(..)
        Di = Differential(t)
        Ii = Integral(t in ClosedInterval(0, t))
        eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(t)) ~ 1
        bcs = [i(0.0) ~ 0.0]
        domains = [t ∈ Interval(0.0, 2.0)]

        chain = Chain(Dense(1, 8, σ), Dense(8, 1))

        @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])

        symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
            pde_system, chain; n_interior = 4, n_bc = 4
        )
        theta0 = symbolic_loss.theta0
        @test isfinite(symbolic_loss.loss(theta0))
        grad = Zygote.gradient(symbolic_loss.loss, theta0)
        @test grad !== nothing
        @test all(isfinite, grad[1])

        discretization_sym = PhysicsInformedNN(
            chain, GridTraining(0.25)
        )
        prob_sym = discretize(pde_system, discretization_sym)

        @test prob_sym isa Optimization.OptimizationProblem

        loss_sym = prob_sym.f(prob_sym.u0, nothing)
        @test isfinite(loss_sym)

        grad_sym = Zygote.gradient(θ -> prob_sym.f(θ, nothing), prob_sym.u0)[1]
        @test grad_sym !== nothing
        @test all(isfinite, grad_sym)

        initial_loss = prob_sym.f(prob_sym.u0, nothing)
        sol = solve(prob_sym, OptimizationOptimisers.Adam(0.01); maxiters = 300)
        final_loss = prob_sym.f(sol.u, nothing)
        @test final_loss < initial_loss
    end

    @testset "2D IDE" begin
        @parameters x, y
        @variables u(..)
        Dx = Differential(x)
        Dy = Differential(y)
        Ix = Integral((x, y) in UnitSquare())

        eq = Ix(u(x, y)) ~ 1 / 3
        bcs = [u(0.0, 0.0) ~ 1.0, Dx(u(x, y)) ~ -2.0 * x, Dy(u(x, y)) ~ -2.0 * y]
        domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

        chain = Chain(Dense(2, 8, σ), Dense(8, 1))

        @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

        symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
            pde_system, chain; n_interior = 4, n_bc = 4
        )
        theta0 = symbolic_loss.theta0
        @test isfinite(symbolic_loss.loss(theta0))
        grad = Zygote.gradient(symbolic_loss.loss, theta0)
        @test grad !== nothing
        @test all(isfinite, grad[1])

        discretization_sym = PhysicsInformedNN(
            chain, GridTraining(0.25)
        )
        prob_sym = discretize(pde_system, discretization_sym)

        @test prob_sym isa Optimization.OptimizationProblem

        loss_sym = prob_sym.f(prob_sym.u0, nothing)
        @test isfinite(loss_sym)

        grad_sym = Zygote.gradient(θ -> prob_sym.f(θ, nothing), prob_sym.u0)[1]
        @test grad_sym !== nothing
        @test all(isfinite, grad_sym)

        initial_loss = prob_sym.f(prob_sym.u0, nothing)
        sol = solve(prob_sym, OptimizationOptimisers.Adam(0.01); maxiters = 300)
        final_loss = prob_sym.f(sol.u, nothing)
        @test final_loss < initial_loss
    end
end

@testitem "IntegroDiff Example 1 -- 1D" tags = [:pinnparser] setup = [IDETestSetup] begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics, NeuralPDE
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters t
    @variables i(..)
    Di = Differential(t)
    Ii = Integral(t in DomainSets.ClosedInterval(0, t))
    eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(t)) ~ 1
    bcs = [i(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chain = Chain(Dense(1, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    analytic_sol_func(t) = 1 / 2 * (exp(-t)) * (sin(2 * t))

    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(phi([t], res.u)) for t in ts]
    @test mean(abs2, u_real .- u_predict) < 0.02
end

@testitem "IntegroDiff Example 3 -- 2 Inputs, 1 Output" tags = [:pinnparser] setup = [IDETestSetup] begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics, NeuralPDE
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x, y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Ix = Integral((x, y) in DomainSets.UnitSquare())

    eq = Ix(u(x, y)) ~ 1 / 3
    bcs = [u(0.0, 0.0) ~ 1, Dx(u(x, y)) ~ -2 * x, Dy(u(x, y)) ~ -2 * y]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(2, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    phi = discretization.phi

    xs = 0.0:0.01:1.0
    ys = 0.0:0.01:1.0

    u_real = collect(1 - x^2 - y^2 for y in ys, x in xs)
    u_predict = collect(Array(phi([x, y], res.u))[1] for y in ys, x in xs)
    @test mean(abs2, u_real .- u_predict) < 0.001
end
