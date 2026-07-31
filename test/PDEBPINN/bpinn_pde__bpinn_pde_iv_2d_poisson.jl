using NeuralPDE
using Test

@testset "BPINN PDE IV: 2D Poisson" begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    # Boundary conditions
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0,
        u(x, 1) ~ 0.0,
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # Discretization
    dt = 0.1f0

    # Neural network
    chain = Chain(Dense(2, 9, σ), Dense(9, 9, σ), Dense(9, 1))

    dx = 0.04
    discretization = BayesianPINN([chain], GridTraining(dx))

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    sol = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 200,
        bcstd = [0.003, 0.003, 0.003, 0.003], phystd = [0.003],
        priorsNNw = (0.0, 10.0), saveats = [1 / 100.0, 1 / 100.0]
    )

    xs = sol.timepoints[1]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)

    u_predict = pmean(sol.ensemblesol[1])
    u_real = [analytic_sol_func(xs[:, i][1], xs[:, i][2]) for i in 1:length(xs[1, :])]
    @test u_predict ≈ u_real rtol = 0.5
end
