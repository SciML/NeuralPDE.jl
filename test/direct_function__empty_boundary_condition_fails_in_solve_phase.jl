using NeuralPDE
using Test

@testset "Empty boundary condition [] fails in solve phase" begin
    using NeuralPDE, Optimization, OptimizationOptimisers, Lux
    @parameters x
    @variables u(..)

    eq = [u(x) ~ 2 + abs(x - 0.5)]
    bc = []
    domain = [x ∈ IntervalDomain(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    for strategy in (
            GridTraining(0.01),
            StochasticTraining(1000),
            QuasiRandomTraining(1000),
            QuadratureTraining(),
        )
        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        prob = discretize(pde_system, discretization)
        @test_throws MethodError solve(prob, Adam(0.05), maxiters = 10)
    end
end
