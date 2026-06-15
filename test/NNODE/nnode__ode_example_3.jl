using NeuralPDE
using Test

@testset "ODE Example 3" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

    linear = (u, p, t) -> [cospi(2t), sinpi(2t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [0.0f0, -1.0f0 / 2pi]
    linear_analytic = (u0, p, t) -> [sinpi(2t) / 2pi, -cospi(2t) / 2pi]
    odefunction = ODEFunction(linear; analytic = linear_analytic)
    prob = ODEProblem(odefunction, u0, tspan)
    luxchain = Chain(Dense(1, 10, σ), Dense(10, 2))
    opt = Adam(0.1)
    alg = NNODE(luxchain, opt; autodiff = false)

    sol = solve(
        prob, alg; verbose = false, maxiters = 1000, abstol = 1.0e-6, saveat = 0.01
    )

    @test sol.errors[:l2] < 0.5
end
