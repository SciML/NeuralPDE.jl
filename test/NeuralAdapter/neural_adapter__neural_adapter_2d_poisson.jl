module NeuralAdapterTestSetup

    function callback(p, l)
        (p.iter == 1 || p.iter % 500 == 0) &&
            println("Current loss is: $l after $(p.iter) iterations")
        return false
    end

    export callback

end

using .NeuralAdapterTestSetup

using NeuralPDE
using Test

@testset "Neural Adapter: 2D Poisson" begin
    using Optimization, Lux, OptimizationOptimisers, Statistics, ComponentArrays, Random,
        LinearAlgebra
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

    # Initial and boundary conditions
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ -sinpi(1) * sinpi(y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sinpi(x) * sinpi(1),
    ]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    quadrature_strategy = QuadratureTraining(
        reltol = 1.0e-3, abstol = 1.0e-6, maxiters = 50, batch = 100
    )
    chain1 = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
    discretization = PhysicsInformedNN(chain1, quadrature_strategy)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(5.0e-3); callback, maxiters = 2000)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)

    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
    u_real = [analytic_sol_func(x, y) for x in xs for y in ys]

    @test u_predict ≈ u_real atol = 5.0e-2 norm = Base.Fix2(norm, Inf)

    chain2 = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
    initp, st = Lux.setup(Random.default_rng(), chain2)
    init_params2 = ComponentArray{Float64}(initp)

    loss(cord, θ) = first(chain2(cord, θ, st)) .- phi(cord, res.u)

    grid_strategy = GridTraining(0.05)
    quadrature_strategy = QuadratureTraining(
        reltol = 1.0e-3, abstol = 1.0e-6, maxiters = 50, batch = 100
    )
    stochastic_strategy = StochasticTraining(1000)
    quasirandom_strategy = QuasiRandomTraining(1000, minibatch = 200, resampling = true)

    @testset "$(nameof(typeof(strategy_)))" for strategy_ in [
            grid_strategy, quadrature_strategy, stochastic_strategy, quasirandom_strategy,
        ]
        prob_ = neural_adapter(loss, init_params2, pde_system, strategy_)
        res_ = solve(prob_, Optimisers.Adam(5.0e-3); callback, maxiters = 1500)
        discretization = PhysicsInformedNN(chain2, strategy_; init_params = res_.u)
        phi_ = discretization.phi

        u_predict_ = [first(phi_([x, y], res_.u)) for x in xs for y in ys]
        @test u_predict_ ≈ u_real atol = 8.0e-2 norm = Base.Fix2(norm, Inf)
    end
end
