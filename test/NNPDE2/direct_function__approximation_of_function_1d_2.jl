using NeuralPDE
using Test

@testset "Approximation of function 1D - 2" begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x
    @variables u(..)
    func(x) = @. cos(5pi * x) * x

    eq = [u(x) ~ func(x)]
    bc = [u(0) ~ u(0)]

    x0 = 0
    x_end = 4
    domain = [x ∈ Interval(x0, x_end)]

    hidden = 20
    chain = Chain(
        Dense(1, hidden, sin), Dense(hidden, hidden, sin),
        Dense(hidden, hidden, sin), Dense(hidden, 1)
    )

    strategy = GridTraining(0.01)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 1000)
    dx = 0.01
    xs = collect(x0:dx:x_end)
    func_s = func(xs)
    @test discretization.phi(xs', res.u) ≈ func(xs') rtol = 0.02
end
