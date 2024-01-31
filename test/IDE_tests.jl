using Test, NeuralPDE
using Optimization, OptimizationOptimJL
import ModelingToolkit: Interval
using DomainSets, Flux
import Lux

using Random
Random.seed!(110)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

@testset "Example 1 - 1D" begin
    @parameters t
    @variables i(..)
    Di = Differential(t)
    Ii = Integral(t in DomainSets.ClosedInterval(0, t))
    eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(t)) ~ 1
    bcs = [i(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 1))
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_)
    @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimJL.BFGS(); callback = callback, maxiters = 100)
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    analytic_sol_func(t) = 1 / 2 * (exp(-t)) * (sin(2 * t))
    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(phi([t], res.minimizer)) for t in ts]
    @test Flux.mse(u_real, u_predict) < 0.01
end

@testset "Example 2 - 1D" begin
    @parameters x
    @variables u(..)
    Ix = Integral(x in DomainSets.ClosedInterval(0, x))
eq = Ix(u(x) * cos(x)) ~ (x^3) / 3
eq = Ix(u(x) * cos(x)) ~ (x^3) / 3

    eq = Ix(u(x) * cos(x)) ~ (x^3) / 3

    bcs = [u(0.0) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.00)]
    chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 1))
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); callback = callback,
                            maxiters = 200)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    u_predict = [first(phi([x], res.minimizer)) for x in xs]
    u_real = [x^2 / cos(x) for x in xs]
    @test Flux.mse(u_real, u_predict) < 0.001
end

@testset "Example 3 - 2 Inputs, 1 Ouput" begin
    @parameters x, y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Ix = Integral((x, y) in DomainSets.UnitSquare())
    eq = Ix(u(x, y)) ~ 1 / 3
    bcs = [u(0.0, 0.0) ~ 1, Dx(u(x, y)) ~ -2 * x, Dy(u(x, y)) ~ -2 * y]
    domains = [x ∈ Interval(0.0, 1.00), y ∈ Interval(0.0, 1.00)]
    chain = Lux.Chain(Lux.Dense(2, 15, Lux.σ), Lux.Dense(15, 1))
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_)
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimJL.BFGS(); callback = callback, maxiters = 100)
    xs = 0.00:0.01:1.00
    ys = 0.00:0.01:1.00
    phi = discretization.phi
    u_real = collect(1 - x^2 - y^2 for y in ys, x in xs);
    u_predict = collect(Array(phi([x, y], res.minimizer))[1] for y in ys, x in xs);
    @test Flux.mse(u_real, u_predict) < 0.001
end

@testset "Example 4 - 2 Inputs, 1 Ouput" begin
    @parameters x, y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Ix = Integral((x, y) in DomainSets.ProductDomain(UnitInterval(), ClosedInterval(0, x)))
    eq = Ix(u(x, y)) ~ 5 / 12
    bcs = [u(0.0, 0.0) ~ 0, Dy(u(x, y)) ~ 2 * y, u(x, 0) ~ x]
    domains = [x ∈ Interval(0.0, 1.00), y ∈ Interval(0.0, 1.00)]
    chain = Lux.Chain(Lux.Dense(2, 15, Lux.σ), Lux.Dense(15, 1))
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_)
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimJL.BFGS(); callback = callback, maxiters = 100)
    xs = 0.00:0.01:1.00
    ys = 0.00:0.01:1.00
    phi = discretization.phi
    u_real = collect(x + y^2 for y in ys, x in xs);
    u_predict = collect(Array(phi([x, y], res.minimizer))[1] for y in ys, x in xs);
    @test Flux.mse(u_real, u_predict) < 0.01
end

@testset "Example 5 - 1 Input, 2 Outputs" begin
    @parameters x
    @variables u(..) w(..)
    Dx = Differential(x)
    Ix = Integral(x in DomainSets.ClosedInterval(1, x))
    eqs = [Ix(u(x) * w(x)) ~ log(abs(x)), Dx(w(x)) ~ -2 / (x^3), u(x) ~ x]
    bcs = [u(1.0) ~ 1.0, w(1.0) ~ 1.0]
    domains = [x ∈ Interval(1.0, 2.0)]
    chains = [Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 1)) for _ in 1:2]
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chains, strategy_)
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x), w(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimJL.BFGS(); callback = callback, maxiters = 200)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    u_predict = [(phi[1]([x], res.u.depvar.u))[1] for x in xs]
    w_predict = [(phi[2]([x], res.u.depvar.w))[1] for x in xs]
    u_real = [x for x in xs]
    w_real = [1 / x^2 for x in xs]
    @test Flux.mse(u_real, u_predict) < 0.001
    @test Flux.mse(w_real, w_predict) < 0.001
end

@testset "Example 6: Infinity" begin
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(1, x))
    Iinf = Integral(x in ClosedInterval(1, Inf))
    eqs = [I(u(x)) ~ Iinf(u(x)) - 1 / x]
    bcs = [u(1) ~ 1]
    domains = [x ∈ Interval(1.0, 2.0)]
    chain = Lux.Chain(Lux.Dense(1, 10, Lux.σ), Lux.Dense(10, 1))
    discretization = PhysicsInformedNN(chain, NeuralPDE.GridTraining(0.1))
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimJL.BFGS(); callback = callback, maxiters = 200)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    u_predict = [first(phi([x], res.minimizer)) for x in xs]
    u_real = [1 / x^2 for x in xs]
    @test u_real≈u_predict rtol=10^-2
end

@testset "Example 7: Infinity" begin
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(x, Inf))
    eq = I(u(x)) ~ 1 / x
    domains = [x ∈ Interval(1.0, 2.0)]
    bcs = [u(1) ~ 1]
    chain = Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimJL.BFGS(); callback = callback, maxiters = 300)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    u_predict = [first(phi([x], res.minimizer)) for x in xs]
    u_real = [1 / x^2 for x in xs]
    @test u_real≈u_predict rtol=10^-2
end
