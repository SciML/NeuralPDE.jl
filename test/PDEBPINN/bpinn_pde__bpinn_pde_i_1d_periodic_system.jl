using NeuralPDE
using Test

@testset "BPINN PDE I: 1D Periodic System" begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t
    @variables u(..)
    Dt = Differential(t)
    eq = Dt(u(t)) - cospi(2t) ~ 0
    bcs = [u(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chainl = Chain(Dense(1, 6, tanh), Dense(6, 1))
    initl, st = Lux.setup(Random.default_rng(), chainl)
    @named pde_system = PDESystem(eq, bcs, domains, [t], [u(t)])

    # non adaptive case
    discretization = BayesianPINN([chainl], GridTraining([0.01]))

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 1500, bcstd = [0.01],
        phystd = [0.01], priorsNNw = (0.0, 1.0), saveats = [1 / 50.0]
    )

    analytic_sol_func(u0, t) = u0 + sinpi(2t) / (2pi)
    ts = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(0.0, t) for t in ts]
    u_predict = pmean(sol1.ensemblesol[1])

    # absol tests
    @test mean(abs, u_predict .- u_real) < 8.0e-2
end
