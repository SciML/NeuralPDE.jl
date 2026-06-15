using NeuralPDE
using Test

@testset "Trivial BC [0 ~ 0] fails for some training strategies" begin
    using NeuralPDE, Optimization, OptimizationOptimisers, Lux
    @parameters x
    @variables u(..)

    eq = [u(x) ~ 2 + abs(x - 0.5)]
    bc = [0 ~ 0]
    domain = [x ∈ IntervalDomain(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    for strategy in (StochasticTraining(1000), QuasiRandomTraining(1000))
        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        @test_throws ArgumentError discretize(pde_system, discretization)
    end
end
