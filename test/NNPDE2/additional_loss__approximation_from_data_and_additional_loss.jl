using NeuralPDE
using Test

@testset "Approximation from data and additional_loss" begin
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

    u_ = (cord, θ, phi) -> sum(phi(cord, θ))

    additional_loss(phi, θ, p) = sum(abs2, phi(xs, θ) .- data)

    discretization = PhysicsInformedNN(chain, strategy; additional_loss)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    res = solve(prob, Adam(0.01); maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 500)
    phi = discretization.phi

    @test phi(xs, res.u) ≈ aproxf(xs) rtol = 0.02
end
