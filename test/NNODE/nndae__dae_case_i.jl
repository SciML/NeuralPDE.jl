using NeuralPDE
using Test

@testset "DAE Case I" begin
    using Random, OrdinaryDiffEq, Statistics, Lux, Optimisers

    Random.seed!(100)

    function example1(du, u, p, t)
        du[1] = cos(2pi * t)
        du[2] = u[2] + cos(2pi * t)
        nothing
    end

    u₀ = [1.0, -1.0]
    du₀ = [0.0, 0.0]
    M = [
        1.0 0.0
        0.0 0.0
    ]
    f = ODEFunction(example1, mass_matrix = M)
    tspan = (0.0f0, 1.0f0)

    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5P(), reltol = 1.0e-8, abstol = 1.0e-8)

    example = (du, u, p, t) -> [cos(2pi * t) - du[1], u[2] + cos(2pi * t) - du[2]]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = [true, false])
    chain = Chain(Dense(1, 15, cos), Dense(15, 15, sin), Dense(15, 2))
    alg = NNDAE(chain, Adam(0.01); autodiff = false)

    sol = solve(
        prob, alg; verbose = false, dt = 1 / 100.0f0, maxiters = 3000, abstol = 1.0f-10
    )

    @test ground_sol(0:(1 / 100):1) ≈ sol atol = 0.4
end
