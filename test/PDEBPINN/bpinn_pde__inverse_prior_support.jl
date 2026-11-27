using AdvancedHMC, Distributions, LogDensityProblems, Lux, MCMCChains,
    ModelingToolkit, NeuralPDE, Random, Test
using DomainSets: Interval

@testset "Bayesian PDE inverse initialization respects prior support" begin
    @parameters x t k
    @variables u(..)
    prior = Beta(2, 2)
    for (k0, pretrain_iters) in ((0.25, 0), (2.0, 3))
        @named sys = PDESystem(
            Differential(t)(u(x, t)) ~ k * x, [u(x, 0.0) ~ 0.0],
            [x ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)], [x, t], [u(x, t)],
            [k]; initial_conditions = Dict(k => k0)
        )
        data = [0.05 0.5 0.2; 0.12 0.8 0.3]
        disc = BayesianPINN(
            Lux.Chain(Lux.Dense(2, 1; use_bias = false)), GridTraining(0.5);
            param_estim = true, dataset = ([data], nothing), rng = Xoshiro(103)
        )
        function fit(phystd)
            Random.seed!(103)
            return ahmc_bayesian_pinn_pde(
                sys, disc; param = [prior], pretrain_iters,
                draw_samples = 12, numensemble = 4, saveats = [0.5, 0.5],
                phystd, bcstd = [0.3], l2std = [0.3]
            )
        end
        sol = fit([0.3])
        coefficients = last.(sol.original.samples)
        @test length(coefficients) == 12
        @test 0 < minimum(coefficients) <= maximum(coefficients) < 1
        @test all(isfinite, logpdf.(prior, coefficients))
        if iszero(pretrain_iters)
            @test_throws "Bayesian PDE initial log density is" fit([0.0])
        end
    end
end
