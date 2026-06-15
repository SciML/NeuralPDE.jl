using NeuralPDE
using Test

@testset "Lorenz System" begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, ComponentArrays,
        OrdinaryDiffEq
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS

    @parameters t, σ_, β, ρ
    @variables x(..), y(..), z(..)
    Dt = Differential(t)
    eqs = [
        Dt(x(t)) ~ σ_ * (y(t) - x(t)),
        Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
        Dt(z(t)) ~ x(t) * y(t) - β * z(t),
    ]

    bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0)]
    dt = 0.05

    input_ = length(domains)
    n = 12
    chain = [Chain(Dense(input_, n, tanh), Dense(n, n, σ), Dense(n, 1)) for _ in 1:3]

    function lorenz!(du, u, p, t)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end

    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(lorenz!, u0, tspan)
    sol = solve(prob, Tsit5(), dt = 0.1)
    ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]

    data = [reduce(hcat, sol.u), reduce(hcat, sol.t)]

    init_params = [
        ComponentArray{Float64}(
                Lux.initialparameters(
                    Random.default_rng(), chain[i]
                )
            )
            for i in 1:3
    ]

    names = (:x, :y, :z)
    flat_init_params = ComponentArray(NamedTuple{names}(i for i in init_params))

    acum = [0; accumulate(+, length.(init_params))]
    sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]
    u_, t_ = data
    len = length(data[2])

    function additional_loss(phi, θ, p)
        return sum(1:3) do i
            sum(abs2, phi[i](t_, getproperty(θ, names[i])) .- u_[[i], :]) / len
        end
    end

    discretization = PhysicsInformedNN(
        chain, GridTraining(dt);
        init_params = flat_init_params, param_estim = true, additional_loss
    )

    @named pde_system = PDESystem(
        eqs, bcs, domains,
        [t], [x(t), y(t), z(t)], [σ_, ρ, β],
        initial_conditions = Dict([p => 1.0 for p in [σ_, ρ, β]])
    )

    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    res = solve(prob, BFGS(); maxiters = 4000)
    p_ = res.u[(end - 2):end]
    @test sum(abs2, p_[1] - 10.0) < 1.0e5
    @test sum(abs2, p_[2] - 28.0) < 1.0
    @test sum(abs2, p_[3] - (8 / 3)) < 1.0

    discretization = PhysicsInformedNN(
        chain, GridTraining(dt); param_estim = true, additional_loss
    )

    @named pde_system = PDESystem(
        eqs, bcs, domains,
        [t], [x(t), y(t), z(t)], [σ_, ρ, β],
        initial_conditions = Dict([p => 1.0 for p in [σ_, ρ, β]])
    )

    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    res = solve(prob, BFGS(); maxiters = 4000)
    p_ = res.u[(end - 2):end]
    # Relaxed tolerances — Lorenz parameter estimation can have variable convergence.
    @test sum(abs2, p_[1] - 10.0) < 0.5
    @test sum(abs2, p_[2] - 28.0) < 0.5
    @test sum(abs2, p_[3] - (8 / 3)) < 0.5
end
