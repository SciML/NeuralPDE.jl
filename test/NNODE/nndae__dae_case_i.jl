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

    # Adam(0.01) on this problem is under-trained at 3000 iters: the algebraic variable
    # u[2] = -cos(2pi t) is still ~0.13 off, leaving the full-grid norm error at ~0.55
    # (above the 0.4 tolerance). The fit converges deterministically with more steps
    # (norm ~0.26 at 10000), so train long enough to land comfortably under tolerance.
    sol = solve(
        prob, alg; verbose = false, dt = 1 / 100.0f0, maxiters = 10000, abstol = 1.0f-10
    )

    @test ground_sol(0:(1 / 100):1) ≈ sol atol = 0.4
end
