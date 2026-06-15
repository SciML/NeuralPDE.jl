using NeuralPDE
using Test

@testset "BPINN PDE: Translating from Flux" begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum
    import Flux

    Random.seed!(100)

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~
        θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0f0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    # Neural network
    chain = Flux.Chain(Flux.Dense(1, 12, Flux.σ), Flux.Dense(12, 1))

    discretization = BayesianPINN([chain], GridTraining([0.01]))
    @test discretization.chain[1] isa Lux.AbstractLuxLayer

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 500,
        bcstd = [0.1], phystd = [0.05], priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol.ensemblesol[1])

    @test u_predict ≈ u_real atol = 0.8
end
