using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "Trivial equation [0 ~ 0] errors in discretize" begin
    using ModelingToolkit, NeuralPDE, SciMLBase, Optimization, OptimizationOptimisers, Lux
    import DomainSets: Interval
    @parameters x
    @variables u(..)

    eq = [0 ~ 0]
    bc = [u(0.0) ~ 2.5]
    domain = [x ∈ Interval(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    strategies = (
        GridTraining(0.01), StochasticTraining(1000),
        QuasiRandomTraining(1000), QuadratureTraining(),
    )

    for strategy in strategies
        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        @test_throws ArgumentError discretize(pde_system, discretization)
    end
end
