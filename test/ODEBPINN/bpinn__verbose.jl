using NeuralPDE, SciMLBase, Lux, AdvancedHMC, MCMCChains, LogDensityProblems, Random, Test

include(joinpath(@__DIR__, "..", "helpers", "verbosity.jl"))

# BNNODE takes its verbosity from `BNNODE(...; verbose)`; the `solve` keyword is not read.
@testset "BNNODE verbose" begin
    prob = ODEProblem((u, p, t) -> cos(2 * π * t), 0.0, (0.0, 1.0))
    bnnode(verbose) = BNNODE(Chain(Dense(1, 4, tanh), Dense(4, 1)); draw_samples = 10, verbose)
    function solve_captured(alg; kw...)
        Random.seed!(100)
        return capture_progress(() -> solve(prob, alg; kw...))
    end
    loglikelihood_printed(c) = occursin("Current Physics Log-likelihood", c.printed)

    c = solve_captured(bnnode(false))
    @test c.result isa BPINNsolution
    @test !loglikelihood_printed(c)

    @test !loglikelihood_printed(solve_captured(bnnode(false); verbose = true))
    @test loglikelihood_printed(solve_captured(bnnode(true); verbose = false))
end
