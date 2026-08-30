using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "Dependent-variable-free boundary conditions" begin
    using ModelingToolkit, NeuralPDE, SciMLBase, Optimization, OptimizationOptimisers, Lux
    import DomainSets: Interval
    @parameters x
    @variables u(..)

    eq = [u(x) ~ 2 + abs(x - 0.5)]
    bc = [0 ~ 0]
    domain = [x ∈ Interval(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    strategies = (
        GridTraining(0.5),
        StochasticTraining(16),
        QuasiRandomTraining(16),
        QuasiRandomTraining(16; resampling = false, minibatch = 2),
        QuadratureTraining(reltol = 0.01, abstol = 0.01, maxiters = 100, batch = 16),
    )

    for strategy in strategies
        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        representation = symbolic_discretize(pde_system, discretization)
        bc_loss = only(representation.loss_functions.bc_loss_functions)(
            representation.flat_init_params
        )
        @test sum(bc_loss) == 0

        prob = discretize(pde_system, discretization)
        @test isfinite(prob.f(prob.u0, prob.p))
        sol = solve(prob, Adam(0.01), maxiters = 2)
        @test isfinite(sol.objective)
    end

    for strategy in strategies
        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, [0 ~ 1], domain, [x], [u(x)])
        representation = symbolic_discretize(pde_system, discretization)
        bc_loss = only(representation.loss_functions.bc_loss_functions)(
            representation.flat_init_params
        )
        @test sum(bc_loss) == 1
    end
end
