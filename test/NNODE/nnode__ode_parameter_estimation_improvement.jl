using NeuralPDE
using Test

@testset "ODE Parameter Estimation Improvement" begin
    using OrdinaryDiffEq, Random, Lux, OptimizationOptimJL, LineSearches
    using FastGaussQuadrature
    Random.seed!(100)

    function lorenz(u, p, t)
        return [
            p[1] * (u[2] - u[1]),
            u[1] * (p[2] - u[3]) - u[2],
            u[1] * u[2] - p[3] * u[3],
        ]
    end
    tspan = (0.0, 5.0)
    prob = ODEProblem(lorenz, [1.0, 0.0, 0.0], tspan, [-10.0, -10.0, -10.0])
    true_p = [2.0, 3.0, 2.0]
    prob2 = remake(prob, p = true_p)
    n = 8
    luxchain = Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, 3))

    # this example is especially easy for the Data Quadrature loss.
    # even with ~2 observed data points, we can exactly calculate the physics parameters (even before solve call).
    N = 7
    # x, w = gausslegendre(N) # does not include endpoints
    x, w = gausslobatto(N)
    # x, w = clenshaw_curtis(N)
    a = tspan[1]
    b = tspan[2]

    # transform the roots and weights
    # x = map((x) -> (2 * (t - a) / (b - a)) - 1, x)
    t = map((x) -> (x * (b - a) + (b + a)) / 2, x)
    W = map((x) -> x * (b - a) / 2, w)
    sol = solve(prob2, Tsit5(); saveat = t)
    t_ = sol.t
    u_ = sol.u
    u1_ = [u_[i][1] for i in eachindex(t_)]
    u2_ = [u_[i][2] for i in eachindex(t_)]
    u3_ = [u_[i][3] for i in eachindex(t_)]
    dataset = [u1_, u2_, u3_, t_, W]

    alg_old = NNODE(
        luxchain, BFGS(linesearch = BackTracking());
        strategy = GridTraining(0.01), dataset = dataset,
        param_estim = true
    )
    sol_old = solve(
        prob, alg_old; verbose = false, abstol = 1.0e-12, maxiters = 2000, saveat = 0.01
    )

    alg_new = NNODE(
        luxchain, BFGS(linesearch = BackTracking()); strategy = GridTraining(0.01),
        param_estim = true, dataset = dataset, estim_collocate = true
    )
    sol_new = solve(
        prob, alg_new; verbose = false, abstol = 1.0e-12, maxiters = 2000, saveat = 0.01
    )

    sol = solve(prob2, Tsit5(); saveat = 0.01)
    sol_points = hcat(sol.u...)
    sol_old_points = hcat(sol_old.u...)
    sol_new_points = hcat(sol_new.u...)

    @test !isapprox(sol_old.k.u.p, true_p; atol = 10)
    @test !isapprox(sol_old_points, sol_points; atol = 10)

    @test sol_new.k.u.p ≈ true_p atol = 5.0e-2
    @test sol_new_points ≈ sol_points atol = 0.2
end
