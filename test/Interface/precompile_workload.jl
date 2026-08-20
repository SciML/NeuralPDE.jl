using ModelingToolkit, NeuralPDE, SciMLBase
using Random
using Test

@testset "Precompile workload" begin
    using Lux, Optimisers
    using DomainSets: Interval

    Random.seed!(2026)
    @parameters x
    @variables u(..)

    @named pde_system = PDESystem(
        Differential(x)(u(x)) ~ 1.0,
        [u(0.0) ~ 0.0],
        [x ∈ Interval(0.0, 1.0)],
        [x],
        [u(x)]
    )
    discretization = PhysicsInformedNN(
        Chain(Dense(1, 2, tanh), Dense(2, 1)), GridTraining(1.0)
    )

    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(0.001); maxiters = 1)

    @test res.u !== nothing
    @test isfinite(res.objective)
end
