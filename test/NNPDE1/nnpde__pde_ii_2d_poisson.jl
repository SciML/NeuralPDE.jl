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

@testset "PDE II: 2D Poisson" begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS
    using LineSearches: BackTracking

    function test_2d_poisson_equation(chain, strategy)
        @parameters x y
        @variables u(..)
        Dxx = Differential(x)^2
        Dyy = Differential(y)^2

        # 2D PDE
        eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

        # Boundary conditions
        bcs = [
            u(0, y) ~ 0.0,
            u(1, y) ~ 0.0,
            u(x, 0) ~ 0.0,
            u(x, 1) ~ 0.0,
        ]

        # Space and time domains
        domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

        ps = Lux.initialparameters(Random.default_rng(), chain)

        discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
        @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
        prob = discretize(pde_system, discretization)
        res = solve(prob, Adam(0.01); maxiters = 1000, callback)
        prob = remake(prob, u0 = res.u)
        res = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 1000)
        phi = discretization.phi

        xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
        analytic = (x, y) -> (sinpi(x) * sinpi(y)) / (2pi^2)

        u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
        u_real = [analytic(x, y) for x in xs for y in ys]

        @test u_predict ≈ u_real atol = 2.0
    end

    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        test_2d_poisson_equation(chain, strategy)
    end

    algs = [CubatureJLp()]
    @testset "$(nameof(typeof(alg)))" for alg in algs
        strategy = QuadratureTraining(
            quadrature_alg = alg, reltol = 1.0e-4,
            abstol = 1.0e-3, maxiters = 30, batch = 10
        )
        test_2d_poisson_equation(chain, strategy)
    end
end
