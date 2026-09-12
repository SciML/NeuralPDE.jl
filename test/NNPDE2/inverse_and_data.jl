include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using OrdinaryDiffEq, Statistics

@testset "parameter estimation for the Lorenz system" begin
    @parameters t σ_ ρ β
    @variables x(..) y(..) z(..)
    Dt = Differential(t)
    eqs = [
        Dt(x(t)) ~ σ_ * (y(t) - x(t)),
        Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
        Dt(z(t)) ~ x(t) * y(t) - β * z(t),
    ]
    bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0)]
    function lorenz!(du, u, p, t)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end
    ode_sol = solve(ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0)), Tsit5(); dt = 0.1)
    ts = 0.0:0.05:1.0
    data = Array(ode_sol(ts))
    function additional_loss(phi, θ, p)
        T = reshape(collect(ts), 1, :)
        return sum(abs2, phi.x(T, θ.x) .- data[1:1, :]) + sum(abs2, phi.y(T, θ.y) .- data[2:2, :]) +
            sum(abs2, phi.z(T, θ.z) .- data[3:3, :])
    end
    chains = [Chain(Dense(1, 12, σ), Dense(12, 12, σ), Dense(12, 1)) for _ in 1:3]
    disc = PhysicsInformedNN(
        chains, GridTraining(0.02); param_estim = true, additional_loss, rng = Xoshiro(9)
    )
    @named pde_system = PDESystem(
        eqs, bcs, domains, [t], [x(t), y(t), z(t)], [σ_, ρ, β];
        initial_conditions = Dict(σ_ => 9.0, ρ => 27.0, β => 3.0)
    )
    sys = symbolic_discretize(pde_system, disc)
    @test length(unknowns(sys)) == 6
    @test length(ModelingToolkit.get_costs(sys)) == 7
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 1000, bfgs_iters = 2000)
    p_est = [sol.original_sol[σ_], sol.original_sol[ρ], sol.original_sol[β]]
    @test isapprox(p_est, [10.0, 28.0, 8 / 3]; rtol = 0.1)
end

@testset "parameters kept fixed and supplied through discretize" begin
    @parameters x a
    @variables u(..)
    Dx = Differential(x)
    eq = Dx(u(x)) ~ a * u(x)
    bcs = [u(0.0) ~ 1.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)], [a])
    chain = Chain(Dense(1, 8, σ), Dense(8, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(10))
    sys = symbolic_discretize(pde_system, disc)
    @test any(isequal(a), ModelingToolkit.parameters(sys))
    prob = discretize(pde_system, disc; p = [a => -1.0])
    sol = train(prob; adam_iters = 300, bfgs_iters = 500)
    xs = 0:0.1:1
    @test maximum(abs, [sol(xi; dv = u(x)) for xi in xs] .- exp.(-xs)) < 0.01
    prob2 = remake(prob; p = [a => 1.0])
    sol2 = train(prob2; adam_iters = 300, bfgs_iters = 500)
    @test maximum(abs, [sol2(xi; dv = u(x)) for xi in xs] .- exp.(xs)) < 0.02
end

@testset "function approximation with an additional data loss" begin
    @parameters x
    @variables u(..)
    func(x) = sinpi(x) + 0.5 * x
    xs = collect(0.0:0.05:1.0)
    ys = func.(xs)
    function additional_loss(phi, θ, p)
        return mean(abs2, vec(phi.u(reshape(xs, 1, :), θ.u)) .- ys)
    end
    eq = [u(x) ~ func(x)]
    bcs = [u(0.0) ~ func(0.0)]
    domains = [x ∈ Interval(0.0, 1.0)]
    chain = Chain(Dense(1, 12, tanh), Dense(12, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1); additional_loss, rng = Xoshiro(11))
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 300, bfgs_iters = 500)
    @test maximum(abs, [sol(xi; dv = u(x)) for xi in xs] .- ys) < 0.01
end
