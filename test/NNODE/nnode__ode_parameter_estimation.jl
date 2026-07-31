using NeuralPDE
using Test

@testset "ODE Parameter Estimation" begin
    using OrdinaryDiffEq, Random, Lux, OptimizationOptimJL, LineSearches
    Random.seed!(100)

    function lorenz(u, p, t)
        return [
            p[1] * (u[2] - u[1]),
            u[1] * (p[2] - u[3]) - u[2],
            u[1] * u[2] - p[3] * u[3],
        ]
    end
    tspan = (0.0, 1.0)
    prob = ODEProblem(lorenz, [1.0, 0.0, 0.0], tspan, [1.0, 1.0, 1.0])
    true_p = [2.0, 3.0, 2.0]
    prob2 = remake(prob, p = true_p)
    n = 8
    luxchain = Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, 3))
    sol = solve(prob2, Tsit5(); saveat = 0.01)
    t_ = sol.t
    u_ = sol.u
    sol_points = hcat(u_...)
    u1_ = [u_[i][1] for i in eachindex(t_)]
    u2_ = [u_[i][2] for i in eachindex(t_)]
    u3_ = [u_[i][3] for i in eachindex(t_)]
    dataset = [u1_, u2_, u3_, t_, ones(length(t_))]

    alg = NNODE(
        luxchain, BFGS(linesearch = BackTracking());
        strategy = GridTraining(0.01), dataset = dataset,
        param_estim = true
    )
    sol = solve(prob, alg; verbose = false, abstol = 1.0e-8, maxiters = 1000, saveat = t_)

    @test sol.k.u.p ≈ true_p atol = 1.0e-2
    @test reduce(hcat, sol.u) ≈ sol_points atol = 1.0e-2
end
