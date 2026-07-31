using NeuralPDE
using Test

@testset "BPINN PDE III: 3rd Degree ODE" begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x
    @variables u(..), Dxu(..), Dxxu(..), O1(..), O2(..)
    Dxxx = Differential(x)^3
    Dx = Differential(x)

    # ODE
    eq = Dx(Dxxu(x)) ~ cospi(x)

    # Initial and boundary conditions
    ep = (cbrt(eps(eltype(Float64))))^2 / 6

    bcs = [
        u(0.0) ~ 0.0,
        u(1.0) ~ cospi(1.0),
        Dxu(1.0) ~ 1.0,
        Dxu(x) ~ Dx(u(x)) + ep * O1(x),
        Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x),
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = [
        Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)),
        Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)),
        Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)),
        Chain(Dense(1, 4, tanh), Dense(4, 1)),
        Chain(Dense(1, 4, tanh), Dense(4, 1)),
    ]

    discretization = BayesianPINN(chain, GridTraining(0.01))

    @named pde_system = PDESystem(
        eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)]
    )

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 200,
        bcstd = [0.01, 0.01, 0.01, 0.01, 0.01], phystd = [0.005],
        priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sinpi(x)) / (π^3)

    u_predict = pmean(sol1.ensemblesol[1])
    xs = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(x) for x in xs]
    @test u_predict ≈ u_real atol = 0.5
end
