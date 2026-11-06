using NeuralPDE, SciMLBase, Lux, Optimisers, StableRNGs, Test
using Logging: Logging

include(joinpath(@__DIR__, "..", "helpers", "verbosity.jl"))

@testset "NNSDE verbose" begin
    prob = SDEProblem((u, p, t) -> u, (u, p, t) -> u, 0.5, (0.0, 1.0))
    chain = Chain(Dense(4, 4, tanh), Dense(4, 1)) |> f64
    function solve_captured(; kw...)
        alg = NNSDE(chain, Adam(0.01); sub_batch = 2, numensemble = 4, rng = StableRNG(1))
        return capture_progress(() -> solve(prob, alg; dt = 0.5, maxiters = 2, kw...))
    end

    @testset "verbose omitted" begin
        c = solve_captured()
        @test c.result isa NeuralPDE.SDEsol
        @test progress_hidden(c)
    end

    @test progress_hidden(solve_captured(verbose = false))

    @testset "verbose = true" begin
        c = solve_captured(verbose = true)
        @test progress_shown(c)
        @test all(r -> r.level == Logging.Info, c.records)
    end
end
