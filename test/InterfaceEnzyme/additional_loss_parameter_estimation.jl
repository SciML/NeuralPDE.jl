using DomainSets: Interval
using LineSearches: BackTracking
using Lux, NeuralPDE, OptimizationOptimJL, OrdinaryDiffEq, Random, Test

@testset "Parameter estimation with three networks and an additional data loss" begin
    Random.seed!(1234)
    @parameters t σ_ ρ β
    @variables x(..) y(..) z(..)
    Dt = Differential(t)
    eqs = [
        Dt(x(t)) ~ σ_ * (y(t) - x(t)),
        Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
        Dt(z(t)) ~ x(t) * y(t) - β * z(t),
    ]
    @named system = PDESystem(
        eqs, [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0], [t ∈ Interval(0.0, 1.0)],
        [t], [x(t), y(t), z(t)], [σ_, ρ, β];
        initial_conditions = Dict(p => 1.0 for p in [σ_, ρ, β])
    )
    function lorenz!(du, u, p, t)
        du[1] = 10 * (u[2] - u[1])
        du[2] = u[1] * (28 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
        return nothing
    end
    times = collect(0.0:0.01:1.0)
    trajectory = solve(ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0)), Tsit5(); dt = 0.1)
    data = Array(trajectory(times))
    points = reshape(times, 1, :)
    function additional_loss(phi, θ, p)
        return (
            sum(abs2, phi.x(points, θ.x) .- data[1:1, :]) +
                sum(abs2, phi.y(points, θ.y) .- data[2:2, :]) +
                sum(abs2, phi.z(points, θ.z) .- data[3:3, :])
        ) / length(times)
    end
    chains = [Chain(Dense(1, 8, σ), Dense(8, 8, σ), Dense(8, 8, σ), Dense(8, 1)) for _ in 1:3]
    disc = PhysicsInformedNN(chains, GridTraining(0.01); param_estim = true, additional_loss)
    prob = discretize(system, disc)
    sol = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 1)
    @test sol.original_sol.stats.gevals > 0
    @test isfinite(sol.original_sol.objective)
end
