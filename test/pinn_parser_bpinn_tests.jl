@testitem "PINN parser: Bayesian PINN 1D Periodic System" tags = [:pinnparser, :pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, AdvancedHMC, LogDensityProblems, Statistics, Random,
        NeuralPDE, MonteCarloMeasurements
    import DomainSets: Interval, ClosedInterval

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

    discretization = BayesianPINN([chainl], GridTraining([0.01]))

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 1500, bcstd = [0.01],
        phystd = [0.01], priorsNNw = (0.0, 1.0), saveats = [1 / 50.0]
    )

    analytic_sol_func(u0, t) = u0 + sinpi(2t) / (2pi)
    ts = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(0.0, t) for t in ts]
    u_predict = pmean(sol1.ensemblesol[1])

    @test mean(abs, u_predict .- u_real) < 8.0e-2
end

@testitem "PINN parser: Bayesian PINN PDE I - 1D Periodic System" tags = [:pinnparser] begin
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
    @named pde_system = PDESystem(eq, bcs, domains, [t], [u(t)])

    discretization = BayesianPINN([chainl], GridTraining([0.01]))

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 1500, bcstd = [0.01],
        phystd = [0.01], priorsNNw = (0.0, 1.0), saveats = [1 / 50.0]
    )

    analytic_sol_func(u0, t) = u0 + sinpi(2t) / (2pi)
    ts = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(0.0, t) for t in ts]
    u_predict = pmean(sol1.ensemblesol[1])

    @test mean(abs, u_predict .- u_real) < 8.0e-2
end

@testitem "PINN parser: Bayesian PINN PDE II - 1D ODE" tags = [:pinnparser] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    eq = Dθ(u(θ)) ~
        θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

    bcs = [u(0.0) ~ 1.0f0]
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    chain = Chain(Dense(1, 12, σ), Dense(12, 1))
    discretization = BayesianPINN([chain], GridTraining([0.01]))
    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 500, bcstd = [0.1],
        phystd = [0.05], priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol1.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol1.ensemblesol[1])
    @test u_predict ≈ u_real atol = 0.8
end

@testitem "PINN parser: Bayesian PINN PDE III - 2D Poisson" tags = [:pinnparser] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0,
        u(x, 1) ~ 0.0,
    ]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

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
