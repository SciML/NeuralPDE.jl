using NeuralPDE, SciMLBase, Lux, Optimisers, StableRNGs, Test
using Logging: NullLogger, with_logger
using SciMLLogging: AbstractVerbosityPreset, All, Detailed, ErrorLevel, MessageLevel,
    Minimal, None, Standard
using WeightInitializers: glorot_uniform

include(joinpath(@__DIR__, "..", "helpers", "verbosity.jl"))

struct ForeignPreset <: AbstractVerbosityPreset end

@testset "training progress capture channels" begin
    stdout_only = capture_progress() do
        println(PROGRESS_MARKER, "\n [NNODE]\tIter: [1/2]\tLoss: 0.5")
    end
    @test !progress_shown(stdout_only, "NNODE"; backend = "logging")
    @test progress_shown(stdout_only, "NNODE"; backend = "core")
end

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
        @test progress_shown(c, "NNODE")
    end

    @testset "NeuralPDEVerbosity" begin
        @test progress_hidden(solve_captured(verbose = NeuralPDE.NeuralPDEVerbosity()))
        @test progress_hidden(solve_captured(verbose = NeuralPDE.NeuralPDEVerbosity(None())))
        @test progress_shown(solve_captured(verbose = NeuralPDE.NeuralPDEVerbosity(Detailed())), "NNODE")
        for preset in (None(), Minimal(), Standard())
            @test progress_hidden(solve_captured(verbose = preset))
        end
        for preset in (Detailed(), All())
            @test progress_shown(solve_captured(verbose = preset), "NNODE")
        end
    end

    @testset "invalid verbosity" begin
        with_logger(NullLogger()) do
            @test_throws ArgumentError(
                "ErrorLevel() throws on emission. Choose a non-throwing message level for training_progress."
            ) solve(
                prob, alg; maxiters = 2,
                verbose = NeuralPDE.NeuralPDEVerbosity(; training_progress = ErrorLevel())
            )
            @test_throws ArgumentError(
                "Unsupported verbosity preset ForeignPreset. Use None(), Minimal(), Standard(), Detailed(), or All()."
            ) solve(prob, alg; maxiters = 2, verbose = ForeignPreset())
            if Sys.WORD_SIZE == 64
                setup_calls = Ref(0)
                function init_weight(rng, dims...)
                    setup_calls[] += 1
                    return glorot_uniform(rng, dims...)
                end
                counted_chain = Chain(Dense(1, 8, σ; init_weight), Dense(8, 1; init_weight))
                counted_alg = NNODE(counted_chain, Adam(0.01))
                for severity in (Int64(typemax(Int32)) + 1, Int64(typemin(Int32)) - 1)
                    @test_throws ArgumentError("training_progress log severity must fit in Int32.") solve(
                        prob, counted_alg; dt = 0.1, maxiters = 2,
                        verbose = NeuralPDE.NeuralPDEVerbosity(; training_progress = MessageLevel(severity))
                    )
                    @test setup_calls[] == 0
                end
            end
        end
    end

    @testset "verbosity does not change training" begin
        init_params, _ = Lux.setup(StableRNG(1), chain)
        trained(verbose) = capture_progress() do
            solve(prob, NNODE(chain, Adam(0.01), deepcopy(init_params)); dt = 0.1, maxiters = 5, verbose)
        end.result
        quiet = trained(false)
        loud = trained(NeuralPDE.NeuralPDEVerbosity(; training_progress = MessageLevel(100)))
        @test quiet.k.u == loud.k.u
        @test quiet.u == loud.u
    end
end
