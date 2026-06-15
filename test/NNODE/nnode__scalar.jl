using NeuralPDE
using Test

@testset "Scalar" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    using OptimizationOptimJL: BFGS

    Random.seed!(100)

    linear = (u, p, t) -> cospi(2t)
    tspan = (0.0f0, 1.0f0)
    u0 = 0.0f0
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
            autodiff in [false, true]

        if autodiff
            @test_throws ArgumentError solve(
                prob, NNODE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0
            )
            continue
        end

        @testset for (dt, abstol) in [(1 / 20.0f0, 1.0e-10), (nothing, 1.0e-6)]
            kwargs = (; verbose = false, dt, abstol, maxiters = 200)
            sol = solve(prob, NNODE(luxchain, opt; autodiff); kwargs...)
        end
    end
end
