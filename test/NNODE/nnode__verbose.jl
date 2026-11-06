using NeuralPDE, SciMLBase, Lux, Optimisers, StableRNGs, Test
using Logging: Logging
using SciMLLogging: Detailed, None

include(joinpath(@__DIR__, "..", "helpers", "verbosity.jl"))

@testset "NNODE verbose" begin
    prob = ODEProblem(ODEFunction((u, p, t) -> @. -u + t), [1.0f0], (0.0f0, 1.0f0))
    chain = Chain(Dense(1, 8, σ), Dense(8, 1))
    alg = NNODE(chain, Adam(0.01))
    solve_captured(; kw...) = capture_progress(() -> solve(prob, alg; maxiters = 2, kw...))

    @testset "verbose omitted" begin
        c = solve_captured()
        @test c.result isa ODESolution
        @test progress_hidden(c)
    end

    @test progress_hidden(solve_captured(verbose = false))

    @testset "verbose = true" begin
        c = solve_captured(verbose = true)
        @test progress_shown(c)
        @test all(r -> r.level == Logging.Info, c.records)
    end

    @testset "NeuralPDEVerbosity" begin
        @test progress_hidden(solve_captured(verbose = NeuralPDE.NeuralPDEVerbosity()))
        @test progress_hidden(solve_captured(verbose = NeuralPDE.NeuralPDEVerbosity(None())))
        @test progress_shown(solve_captured(verbose = NeuralPDE.NeuralPDEVerbosity(Detailed())))
        @test progress_shown(solve_captured(verbose = Detailed()))
    end

    @testset "verbosity does not change training" begin
        init_params, _ = Lux.setup(StableRNG(1), chain)
        trained(verbose) = capture_progress() do
            solve(prob, NNODE(chain, Adam(0.01), deepcopy(init_params)); dt = 0.1, maxiters = 5, verbose)
        end.result
        quiet, loud = trained(false), trained(true)
        @test quiet.k.u == loud.k.u
        @test quiet.u == loud.u
    end
end
