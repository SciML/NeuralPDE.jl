using NeuralPDE
using Test

@testset "DAE Case II" begin
    using Random, OrdinaryDiffEq, Statistics, Lux, Optimisers

    Random.seed!(100)

    function example2(du, u, p, t)
        du[1] = u[1] - t
        du[2] = u[2] - t
        nothing
    end

    M = [
        0.0 0.0
        0.0 1.0
    ]
    u₀ = [0.0, 0.0]
    du₀ = [0.0, 0.0]
    tspan = (0.0f0, pi / 2.0f0)
    f = ODEFunction(example2, mass_matrix = M)
    prob_mm = ODEProblem(f, u₀, tspan)
    ground_sol = solve(prob_mm, Rodas5P(), reltol = 1.0e-8, abstol = 1.0e-8)

    example = (du, u, p, t) -> [u[1] - t - du[1], u[2] - t - du[2]]
    differential_vars = [false, true]
    prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
    chain = Chain(Dense(1, 15, σ), Dense(15, 2))
    alg = NNDAE(chain, Adam(0.1); autodiff = false)

    sol = solve(
        prob,
        alg, verbose = false, dt = 1 / 100.0f0, maxiters = 3000, abstol = 1.0f-10
    )

    @test ground_sol(0:(1 / 100):(pi / 2)) ≈ sol atol = 0.4
end
