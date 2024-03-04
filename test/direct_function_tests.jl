using NeuralPDE, Test
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
import Lux

Random.seed!(110)

@testset "Approximation of function 1D" begin
    @parameters x
    @variables u(..)

    func(x) = @. 2 + abs(x - 0.5)

    eq = [u(x) ~ func(x)]
    bc = [u(0) ~ u(0)]

    x0 = 0
    x_end = 2
    dx = 0.001
    domain = [x ∈ Interval(x0, x_end)]

    xs = collect(x0:dx:x_end)
    func_s = func(xs)

    hidden = 10
    chain = Lux.Chain(Lux.Dense(1, hidden, Lux.tanh),
                    Lux.Dense(hidden, hidden, Lux.tanh),
                    Lux.Dense(hidden, 1))

    strategy = GridTraining(0.01)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(initial_stepnorm = 0.01), maxiters = 500)
    @test discretization.phi(xs', res.u)≈func(xs') rtol=0.01
end

@testset "Approximation of function 1D - 2" begin
    @parameters x
    @variables u(..)
    func(x) = @. cos(5pi * x) * x
    eq = [u(x) ~ func(x)]
    bc = [u(0) ~ u(0)]

    x0 = 0
    x_end = 4
    domain = [x ∈ Interval(x0, x_end)]

    hidden = 20
    chain = Lux.Chain(Lux.Dense(1, hidden, Lux.sin),
                    Lux.Dense(hidden, hidden, Lux.sin),
                    Lux.Dense(hidden, hidden, Lux.sin),
                    Lux.Dense(hidden, 1))

    strategy = GridTraining(0.01)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)
    dx = 0.01
    xs = collect(x0:dx:x_end)
    func_s = func(xs)
    @test discretization.phi(xs', res.u)≈func(xs') rtol=0.01
end

@testset "Approximation of function 2D" begin
    @parameters x, y
    @variables u(..)
    func(x, y) = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))
    eq = [u(x, y) ~ func(x, y)]
    bc = [u(0, 0) ~ u(0, 0)]
    x0 = -10
    x_end = 10
    y0 = -10
    y_end = 10
    d = 0.4
    domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]
    hidden = 25
    chain = Lux.Chain(Lux.Dense(2, hidden, Lux.tanh),
                    Lux.Dense(hidden, hidden, Lux.tanh),
                    Lux.Dense(hidden, hidden, Lux.tanh),
                    Lux.Dense(hidden, 1))

    strategy = GridTraining(d)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    symprob.loss_functions.full_loss_function(symprob.flat_init_params, nothing)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(), maxiters = 500)
    phi = discretization.phi
    xs = collect(x0:0.1:x_end)
    ys = collect(y0:0.1:y_end)
    u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
                        (length(xs), length(ys)))
    u_real = reshape([func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
    diff_u = abs.(u_predict .- u_real)
    @test u_predict≈u_real rtol=0.05
end
