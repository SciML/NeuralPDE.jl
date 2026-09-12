using PrecompileTools: @compile_workload, @setup_workload
using DomainSets: Interval
using Lux: Chain, Dense

@setup_workload begin
    @parameters x
    @variables u(..)
    Random.seed!(2026)

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

    @compile_workload begin
        # ForwardDiff keeps the workload cheap; the Enzyme path is compiled on first use.
        prob = discretize(pde_system, discretization; adtype = AutoForwardDiff())
        solve(prob, Adam(0.001); maxiters = 1)
    end
end
