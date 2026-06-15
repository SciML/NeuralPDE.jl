using NeuralPDE
using Test

@testset "ODE Complex Numbers" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

    function bloch_equations(u, p, t)
        Ω, Δ, Γ = p
        γ = Γ / 2
        ρ₁₁, ρ₂₂, ρ₁₂, ρ₂₁ = u
        d̢ρ = [
            im * Ω * (ρ₁₂ - ρ₂₁) + Γ * ρ₂₂;
            -im * Ω * (ρ₁₂ - ρ₂₁) - Γ * ρ₂₂;
            -(γ + im * Δ) * ρ₁₂ - im * Ω * (ρ₂₂ - ρ₁₁);
            conj(-(γ + im * Δ) * ρ₁₂ - im * Ω * (ρ₂₂ - ρ₁₁))
        ]
        return d̢ρ
    end

    u0 = zeros(ComplexF64, 4)
    u0[1] = 1
    time_span = (0.0, 2.0)
    parameters = [2.0, 0.0, 1.0]

    problem = ODEProblem(bloch_equations, u0, time_span, parameters)

    chain = Chain(
        Dense(1, 16, tanh; init_weight = kaiming_normal(ComplexF64)),
        Dense(16, 4; init_weight = kaiming_normal(ComplexF64))
    )
    ps, st = Lux.setup(Random.default_rng(), chain)

    ground_truth = solve(problem, Tsit5(), saveat = 0.01)

    strategies = [
        StochasticTraining(500),
        GridTraining(0.01),
        WeightedIntervalTraining([0.1, 0.4, 0.4, 0.1], 500),
    ]

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        alg = NNODE(chain, Adam(0.01); strategy)
        sol = solve(problem, alg; verbose = false, maxiters = 10000, saveat = 0.01)
        @test sol.u ≈ ground_truth.u rtol = 2.0e-1
    end

    alg = NNODE(chain, Adam(0.01); strategy = QuadratureTraining())
    @test_throws ErrorException solve(
        problem, alg; verbose = false, maxiters = 5000, saveat = 0.01
    )
end
