@testitem "Test-1" tags=[:nnsde] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    using OptimizationOptimJL: BFGS
    Random.seed!(100)

    α = 1
    β = 1
    u₀ = 1 / 2
    f(u, p, t) = α * u
    g(u, p, t) = β * t
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u₀, tspan)
    dim = 1 + 3
    luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
        autodiff in [false, true]

        if autodiff
            @test_throws ArgumentError solve(
                prob, NNSDE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0)
            continue
        end

        @testset for (dt, abstol) in [(1 / 20.0f0, 1e-10), (nothing, 1e-6)]
            kwargs = (; verbose = false, dt, abstol, maxiters = 200)
            sol = solve(prob, NNSDE(luxchain, opt; autodiff); kwargs...)
        end
    end
end