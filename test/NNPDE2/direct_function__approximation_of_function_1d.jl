using NeuralPDE
using Test

@testset "Approximation of function 1D" begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

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
    chain = Chain(Dense(1, hidden, tanh), Dense(hidden, hidden, tanh), Dense(hidden, 1))

    strategy = GridTraining(0.01)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(0.05), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(initial_stepnorm = 0.01), maxiters = 500)

    @test discretization.phi(xs', res.u) ≈ func(xs') rtol = 0.02
end
