using NeuralPDE, SciMLBase, Lux, Optimisers, Test
using Logging: Logging

include(joinpath(@__DIR__, "..", "helpers", "verbosity.jl"))

@testset "NNDAE verbose" begin
    example = (du, u, p, t) -> [cos(2pi * t) - du[1], u[2] + cos(2pi * t) - du[2]]
    prob = DAEProblem(
        example, [0.0, 0.0], [1.0, -1.0], (0.0, 1.0); differential_vars = [true, false]
    )
    alg = NNDAE(Chain(Dense(1, 8, cos), Dense(8, 2)), Adam(0.01); autodiff = false)
    solve_captured(; kw...) = capture_progress(() -> solve(prob, alg; dt = 1 / 10, maxiters = 2, kw...))

    @testset "verbose omitted" begin
        c = solve_captured()
        @test c.result isa SciMLBase.AbstractDAESolution
        @test progress_hidden(c)
    end

    @test progress_hidden(solve_captured(verbose = false))

    @testset "verbose = true" begin
        c = solve_captured(verbose = true)
        @test progress_shown(c)
        @test all(r -> r.level == Logging.Info, c.records)
    end
end
