using NeuralPDE
using Test

@testset "BPINN PDE Inv I: 1D Periodic System" begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t p
    @variables u(..)

    Dt = Differential(t)
    eqs = Dt(u(t)) - cos(p * t) ~ 0
    bcs = [u(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
    initl, st = Lux.setup(Random.default_rng(), chainl)

    @named pde_system = PDESystem(
        eqs,
        bcs,
        domains,
        [t],
        [u(t)],
        [p],
        initial_conditions = Dict([p => 4.0])
    )

    analytic_sol_func1(u0, t) = u0 + sinpi(2t) / (2π)
    timepoints = collect(0.0:(1 / 100.0):2.0)
    u = [analytic_sol_func1(0.0, timepoint) for timepoint in timepoints]
    u = u .+ (u .* 0.2) .* randn(size(u))
    dataset = [hcat(u, timepoints)]

    # BPINNs are formulated with a mesh that must stay the same throughout sampling (as of now)
    @testset "$(nameof(typeof(strategy)))" for strategy in [
            # StochasticTraining(200),
            # QuasiRandomTraining(200),
            GridTraining([0.02]),
        ]
        discretization = BayesianPINN(
            [chainl], strategy; param_estim = true,
            dataset = [dataset, nothing]
        )

        sol1 = ahmc_bayesian_pinn_pde(
            pde_system,
            discretization;
            draw_samples = 1500,
            bcstd = [0.02],
            phystd = [0.02], l2std = [0.02],
            priorsNNw = (0.0, 1.0),
            saveats = [1 / 50.0],
            param = [LogNormal(6.0, 0.5)]
        )

        param = 2 * π
        ts = vec(sol1.timepoints[1])
        u_real = [analytic_sol_func1(0.0, t) for t in ts]
        u_predict = pmean(sol1.ensemblesol[1])

        @test mean(abs, u_predict .- u_real) < 8.0e-2
        @test sol1.estimated_de_params[1] ≈ param rtol = 0.1
    end
end
