module NNPDE1TestSetup

    using NeuralPDE, Cubature, Integrals, QuasiMonteCarlo

    # DataGen is Real: https://github.com/SciML/NeuralPDE.jl/issues/906
    @parameters x
    @variables u(..)

    NeuralPDE.generate_training_sets(
        [x ∈ (-1.0, 1.0)], 0.1, [u(x) ~ x], [0.0 ~ 0.0], Float64, [x], [:u]
    )

    function callback(p, l)
        if p.iter == 1 || p.iter % 250 == 0
            println("Current loss is: $l after $(p.iter) iterations")
        end
        return false
    end

    grid_strategy = GridTraining(0.1)
    quadrature_strategy = QuadratureTraining(
        quadrature_alg = CubatureJLh(),
        reltol = 1.0e3, abstol = 1.0e-3, maxiters = 50, batch = 100
    )
    stochastic_strategy = StochasticTraining(100; bcs_points = 50)
    quasirandom_strategy = QuasiRandomTraining(
        100; sampling_alg = LatinHypercubeSample(),
        resampling = false, minibatch = 100
    )
    quasirandom_strategy_resampling = QuasiRandomTraining(
        100; bcs_points = 50,
        sampling_alg = LatticeRuleSample(), resampling = true, minibatch = 0
    )

    strategies = [
        grid_strategy,
        stochastic_strategy,
        quasirandom_strategy,
        quasirandom_strategy_resampling,
        quadrature_strategy,
    ]

    export callback, strategies

end

using .NNPDE1TestSetup

using NeuralPDE
using Test

@testset "PDE VI: PDE with mixed derivative" begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS
    using LineSearches: BackTracking

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dx = Differential(x)
    Dy = Differential(y)

    eq = Dxx(u(x, y)) + Dx(Dy(u(x, y))) - 2 * Dyy(u(x, y)) ~ -1.0

    # Initial and boundary conditions
    bcs = [
        u(x, 0) ~ x,
        Dy(u(x, 0)) ~ x,
        u(x, 0) ~ Dy(u(x, 0)),
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    strategy = StochasticTraining(2048)
    inner = 32
    chain = Chain(Dense(2, inner, sigmoid), Dense(inner, inner, sigmoid), Dense(inner, 1))

    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); maxiters = 500, callback)
    phi = discretization.phi

    analytic_sol_func(x, y) = x + x * y + y^2 / 2
    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
    u_real = [analytic_sol_func(x, y) for x in xs for y in ys]
    @test u_predict ≈ u_real rtol = 0.1
end
