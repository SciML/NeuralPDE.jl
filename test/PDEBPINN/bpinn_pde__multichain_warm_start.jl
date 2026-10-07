using NeuralPDE, ModelingToolkit, SciMLBase, Lux, Random, Test
using AdvancedHMC, MCMCChains, LogDensityProblems
using DomainSets: Interval

@testset "Each Bayesian PDE chain is warm-started" begin
    @parameters t
    @variables u(..)
    Dt = Differential(t)
    @named sys = PDESystem(
        Dt(u(t)) ~ 1.0, [u(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [u(t)]
    )
    disc = BayesianPINN(
        Lux.Chain(Lux.Dense(1, 1)), GridTraining(0.5); rng = Xoshiro(100)
    )
    Random.seed!(100)
    solutions, output = mktemp() do _, io
        solutions = redirect_stdout(io) do
            ahmc_bayesian_pinn_pde(
                sys, disc; nchains = 2, draw_samples = 12, numensemble = 4,
                pretrain_iters = 3, verbose = true, saveats = [0.5]
            )
        end
        flush(io)
        seekstart(io)
        return solutions, read(io, String)
    end
    @test count("Pretrain objective after 3 Adam steps:", output) == length(solutions) == 2
    @test all(sol -> length(sol.original.samples) == 12, solutions)
end
