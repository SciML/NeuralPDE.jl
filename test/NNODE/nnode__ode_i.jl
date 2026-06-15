using NeuralPDE
using Test

@testset "ODE I" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    linear = (
        u,
        p,
        t,
    ) -> @. t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
        u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))
    linear_analytic = (u0, p, t) -> [exp(-(t^2) / 2) / (1 + t + t^3) + t^2]
    prob = ODEProblem(
        ODEFunction(linear; analytic = linear_analytic), [1.0f0], (0.0f0, 1.0f0)
    )
    luxchain = Chain(Dense(1, 128, σ), Dense(128, 1))
    opt = Adam(0.01)

    @testset for strategy in [nothing, StochasticTraining(100)], batch in [false, true]

        sol = solve(
            prob, NNODE(luxchain, opt; batch, strategy); verbose = false, maxiters = 200,
            abstol = 1.0e-6
        )
        @test sol.errors[:l2] < 0.5
    end
end
