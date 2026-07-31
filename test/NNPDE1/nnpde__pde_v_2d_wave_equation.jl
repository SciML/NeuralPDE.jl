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

@testset "PDE V: 2D Wave Equation" begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals,
        LineSearches, Integrals
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    # Seed for reproducibility — PINNs can get stuck in bad local minima without it.
    Random.seed!(100)

    @parameters x, t
    @variables u(..)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    Dt = Differential(t)

    # 2D PDE
    C = 1
    eq = Dtt(u(x, t)) ~ C^2 * Dxx(u(x, t))

    # Initial and boundary conditions
    bcs = [
        u(0, t) ~ 0.0,           # for all t > 0
        u(1, t) ~ 0.0,           # for all t > 0
        u(x, 0) ~ x * (1.0 - x), # for all 0 < x < 1
        Dt(u(x, 0)) ~ 0.0,        # for all  0 < x < 1]
    ]

    # Space and time domains
    domains = [
        x ∈ Interval(0.0, 1.0),
        t ∈ Interval(0.0, 1.0),
    ]
    @named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    # Neural network
    chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
    phi = NeuralPDE.Phi(chain)
    derivative = NeuralPDE.numeric_derivative

    quadrature_strategy = QuadratureTraining(
        quadrature_alg = CubatureJLh(),
        reltol = 1.0e-3, abstol = 1.0e-3, maxiters = 50, batch = 100
    )

    discretization = PhysicsInformedNN(chain, quadrature_strategy)
    prob = discretize(pde_system, discretization)

    cb_ = function (p, l)
        println("loss: ", l)
        println("losses: ", map(l -> l(p.u), loss_functions))
        return false
    end

    # Adam warmup for robustness, then BFGS for convergence
    res = solve(prob, Adam(0.01); maxiters = 2000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 2000)

    dx = 0.1
    xs, ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    function analytic_sol_func(x, t)
        sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50000])
    end

    u_predict = reshape(
        [first(phi([x, t], res.u)) for x in xs for t in ts],
        (length(xs), length(ts))
    )
    u_real = reshape(
        [analytic_sol_func(x, t) for x in xs for t in ts],
        (length(xs), length(ts))
    )
    @test u_predict ≈ u_real atol = 0.5
end
