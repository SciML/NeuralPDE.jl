using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "Empty boundary conditions" begin
    using ModelingToolkit, NeuralPDE, SciMLBase, Optimization, OptimizationOptimisers, Lux
    import DomainSets: Interval
    @parameters x
    @variables u(..)

    eq = [u(x) ~ 2 + abs(x - 0.5)]
    bc = []
    domain = [x ∈ Interval(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    for strategy in (
            GridTraining(0.5),
            StochasticTraining(16),
            QuasiRandomTraining(16),
            QuasiRandomTraining(16; resampling = false, minibatch = 2),
            QuadratureTraining(
                reltol = 0.01, abstol = 0.01, maxiters = 100, batch = 16
            ),
        )
        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        prob = discretize(pde_system, discretization)
        @test isfinite(prob.f(prob.u0, prob.p))
        sol = solve(prob, Adam(0.01), maxiters = 2)
        @test isfinite(sol.objective)
    end
end
