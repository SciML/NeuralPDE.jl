using NeuralPDE
using Test

@testset "Vector" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    using OptimizationOptimJL: BFGS

    Random.seed!(100)

    linear = (u, p, t) -> [cospi(2t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0]
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
            autodiff in [false, true]

        sol = solve(
            prob, NNODE(luxchain, opt); verbose = false, maxiters = 200, abstol = 1.0e-6
        )

        @test sol(0.5) isa Vector
        @test sol(0.5; idxs = 1) isa Number
        @test sol.k isa SciMLBase.OptimizationSolution
    end
end
