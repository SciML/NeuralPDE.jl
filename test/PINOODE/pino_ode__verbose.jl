using NeuralPDE, SciMLBase, Lux, Optimisers, StableRNGs, Test
using Logging: Logging

include(joinpath(@__DIR__, "..", "helpers", "verbosity.jl"))

@testset "PINOODE verbose" begin
    prob = ODEProblem((u, p, t) -> cos(p * t), 1.0, (0.0, 1.0))
    chain = Chain(Dense(2 => 8, tanh), Dense(8 => 1))
    function solve_captured(; kw...)
        alg = PINOODE(
            chain, Adam(0.01), [(pi, 2pi)], 10;
            strategy = StochasticTraining(10), rng = StableRNG(1)
        )
        return capture_progress(() -> solve(prob, alg; maxiters = 2, kw...))
    end

    @testset "verbose omitted" begin
        c = solve_captured()
        @test c.result isa SciMLBase.PDETimeSeriesSolution
        @test progress_hidden(c)
    end

    @test progress_hidden(solve_captured(verbose = false))

    @testset "verbose = true" begin
        c = solve_captured(verbose = true)
        @test progress_shown(c)
        @test all(r -> r.level == Logging.Info, c.records)
    end
end
