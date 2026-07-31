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

@testset "PDE I: Heterogeneous system" begin
    using DomainSets, Lux, Random, Optimisers, Integrals
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    @parameters x, y, z
    @variables u(..), v(..), h(..), p(..)
    Dz = Differential(z)
    eqs = [
        u(x, y, z) ~ x + y + z,
        v(y, x) ~ x^2 + y^2,
        h(z) ~ cos(z),
        p(x, z) ~ exp(x) * exp(z),
        u(x, y, z) + v(y, x) * Dz(h(z)) - p(x, z) ~ x + y + z - (x^2 + y^2) * sin(z) -
            exp(x) * exp(z),
    ]

    bcs = [u(0.0, 0.0, 0.0) ~ 0.0]

    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)]

    chain = [
        Chain(Dense(3, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
        Chain(Dense(2, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
        Chain(Dense(1, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
        Chain(Dense(2, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
    ]

    grid_strategy = GridTraining(0.1)
    quadrature_strategy = QuadratureTraining(
        quadrature_alg = CubatureJLh(),
        reltol = 1.0e-3, abstol = 1.0e-3, maxiters = 50, batch = 100
    )

    discretization = PhysicsInformedNN(chain, grid_strategy)

    @named pde_system = PDESystem(
        eqs, bcs, domains, [x, y, z],
        [u(x, y, z), v(y, x), h(z), p(x, z)]
    )

    prob = discretize(pde_system, discretization)

    res = solve(prob, BFGS(); maxiters = 2000, callback)

    phi = discretization.phi

    analytic_sol_func_ = [
        (x, y, z) -> x + y + z,
        (x, y) -> x^2 + y^2,
        (z) -> cos(z),
        (x, z) -> exp(x) * exp(z),
    ]

    xs, ys, zs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

    u_real = [analytic_sol_func_[1](x, y, z) for x in xs for y in ys for z in zs]
    v_real = [analytic_sol_func_[2](y, x) for y in ys for x in xs]
    h_real = [analytic_sol_func_[3](z) for z in zs]
    p_real = [analytic_sol_func_[4](x, z) for x in xs for z in zs]

    real_ = [u_real, v_real, h_real, p_real]

    u_predict = [
        phi[1]([x, y, z], res.u.depvar.u)[1] for x in xs for y in ys
            for z in zs
    ]
    v_predict = [phi[2]([y, x], res.u.depvar.v)[1] for y in ys for x in xs]
    h_predict = [phi[3]([z], res.u.depvar.h)[1] for z in zs]
    p_predict = [phi[4]([x, z], res.u.depvar.p)[1] for x in xs for z in zs]

    predict = [u_predict, v_predict, h_predict, p_predict]

    for i in 1:4
        @test predict[i] ≈ real_[i] rtol = 10^-2
    end
end
