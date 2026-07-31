using NeuralPDE
using Test

@testset "Training Strategy: Others" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, Integrals

    Random.seed!(100)

    linear = (u, p, t) -> cospi(2t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sinpi(2t)
    tspan = (0.0, 1.0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(ODEFunction(linear; analytic = linear_analytic), 0.0, (0.0, 1.0))
    opt = Adam(0.1, (0.9, 0.95))
    u_analytical(x) = (1 / (2pi)) .* sinpi.(2x)

    luxchain = Chain(Dense(1, 5, σ), Dense(5, 1))
    u_, t_ = u_analytical(ts), ts

    function additional_loss(phi, θ)
        return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
    end

    @testset "$(nameof(typeof(strategy)))" for strategy in [
            GridTraining(0.01),
            StochasticTraining(1000),
            QuadratureTraining(
                reltol = 1.0e-3, abstol = 1.0e-6, maxiters = 50,
                batch = 100, quadrature_alg = QuadGKJL()
            ),
        ]
        alg = NNODE(luxchain, opt; additional_loss, strategy)
        @test begin
            sol = solve(prob, alg; verbose = false, maxiters = 500, abstol = 1.0e-6)
            sol.errors[:l2] < 0.5
        end
    end
end
