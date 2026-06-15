using NeuralPDE
using Test

@testset "ODE Example 2" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

    linear = (u, p, t) -> -u / 5 + exp(-t / 5) .* cos(t)
    linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
    prob = ODEProblem(
        ODEFunction(linear; analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0)
    )
    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))

    @testset for batch in [false, true], strategy in [StochasticTraining(100), nothing]

        opt = Adam(0.1)
        sol = solve(
            prob, NNODE(luxchain, opt; batch, strategy); verbose = false, maxiters = 200,
            abstol = 1.0e-6
        )
        @test sol.errors[:l2] < 0.5
    end
end
