@testsetup module NNPDE1TestSetup

using NeuralPDE, Cubature, Integrals, QuasiMonteCarlo

# DataGen is Real: https://github.com/SciML/NeuralPDE.jl/issues/906
@parameters x
@variables u(..)

NeuralPDE.generate_training_sets(
    [x ∈ (-1.0, 1.0)], 0.1, [u(x) ~ x], [0.0 ~ 0.0], Float64, [x], [:u])

function callback(p, l)
    if p.iter == 1 || p.iter % 250 == 0
        println("Current loss is: $l after $(p.iter) iterations")
    end
    return false
end

grid_strategy = GridTraining(0.1)
quadrature_strategy = QuadratureTraining(quadrature_alg = CubatureJLh(),
    reltol = 1e3, abstol = 1e-3, maxiters = 50, batch = 100)
stochastic_strategy = StochasticTraining(100; bcs_points = 50)
quasirandom_strategy = QuasiRandomTraining(100; sampling_alg = LatinHypercubeSample(),
    resampling = false, minibatch = 100)
quasirandom_strategy_resampling = QuasiRandomTraining(100; bcs_points = 50,
    sampling_alg = LatticeRuleSample(), resampling = true, minibatch = 0)

strategies = [
    grid_strategy,
    stochastic_strategy,
    quasirandom_strategy,
    quasirandom_strategy_resampling,
    quadrature_strategy
]

export callback, strategies

end

@testitem "Test Heterogeneous ODE" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
    using Cubature, Integrals, QuasiMonteCarlo, DomainSets, Lux, Random, Optimisers

    function simple_1d_ode(strategy)
        @parameters θ
        @variables u(..)
        Dθ = Differential(θ)

        # 1D ODE
        eq = Dθ(u(θ)) ~ θ^3 + 2.0f0 * θ +
                        (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
                        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

        # Initial and boundary conditions
        bcs = [u(0.0) ~ 1.0f0]

        # Space and time domains
        domains = [θ ∈ Interval(0.0f0, 1.0f0)]

        # Neural network
        chain = Chain(Dense(1, 12, σ), Dense(12, 1))

        discretization = PhysicsInformedNN(chain, strategy)
        @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])
        prob = discretize(pde_system, discretization)

        res = solve(prob, Adam(0.1); maxiters = 1000)
        prob = remake(prob, u0 = res.u)
        res = solve(prob, Adam(0.01); maxiters = 500)
        prob = remake(prob, u0 = res.u)
        res = solve(prob, Adam(0.001); maxiters = 500)
        phi = discretization.phi

        analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
        ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
        u_real = [analytic_sol_func(t) for t in ts]
        u_predict = [first(phi([t], res.u)) for t in ts]
        @test u_predict≈u_real atol=0.8
    end

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        simple_1d_ode(strategy)
    end
end

@testitem "PDE I: Heterogeneous system" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
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
                                                    exp(x) * exp(z)
    ]

    bcs = [u(0.0, 0.0, 0.0) ~ 0.0]

    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)]

    chain = [
        Chain(Dense(3, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
        Chain(Dense(2, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
        Chain(Dense(1, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)),
        Chain(Dense(2, 12, tanh), Dense(12, 12, tanh), Dense(12, 1))
    ]

    grid_strategy = GridTraining(0.1)
    quadrature_strategy = QuadratureTraining(quadrature_alg = CubatureJLh(),
        reltol = 1e-3, abstol = 1e-3, maxiters = 50, batch = 100)

    discretization = PhysicsInformedNN(chain, grid_strategy)

    @named pde_system = PDESystem(eqs, bcs, domains, [x, y, z],
        [u(x, y, z), v(y, x), h(z), p(x, z)])

    prob = discretize(pde_system, discretization)

    res = solve(prob, BFGS(); maxiters = 2000, callback)

    phi = discretization.phi

    analytic_sol_func_ = [
        (x, y, z) -> x + y + z,
        (x, y) -> x^2 + y^2,
        (z) -> cos(z),
        (x, z) -> exp(x) * exp(z)
    ]

    xs, ys, zs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

    u_real = [analytic_sol_func_[1](x, y, z) for x in xs for y in ys for z in zs]
    v_real = [analytic_sol_func_[2](y, x) for y in ys for x in xs]
    h_real = [analytic_sol_func_[3](z) for z in zs]
    p_real = [analytic_sol_func_[4](x, z) for x in xs for z in zs]

    real_ = [u_real, v_real, h_real, p_real]

    u_predict = [phi[1]([x, y, z], res.u.depvar.u)[1] for x in xs for y in ys
                 for z in zs]
    v_predict = [phi[2]([y, x], res.u.depvar.v)[1] for y in ys for x in xs]
    h_predict = [phi[3]([z], res.u.depvar.h)[1] for z in zs]
    p_predict = [phi[4]([x, z], res.u.depvar.p)[1] for x in xs for z in zs]

    predict = [u_predict, v_predict, h_predict, p_predict]

    for i in 1:4
        @test predict[i]≈real_[i] rtol=10^-2
    end
end

@testitem "PDE II: 2D Poisson" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum

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
            u(x, 1) ~ 0.0
        ]

        # Space and time domains
        domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

        ps = Lux.initialparameters(Random.default_rng(), chain)

        discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
        @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
        prob = discretize(pde_system, discretization)
        res = solve(prob, Adam(0.1); maxiters = 500, callback)
        phi = discretization.phi

        xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
        analytic = (x, y) -> (sinpi(x) * sinpi(y)) / (2pi^2)

        u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
        u_real = [analytic(x, y) for x in xs for y in ys]

        @test u_predict≈u_real atol=2.0
    end

    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        test_2d_poisson_equation(chain, strategy)
    end

    algs = [CubatureJLp()]
    @testset "$(nameof(typeof(alg)))" for alg in algs
        strategy = QuadratureTraining(quadrature_alg = alg, reltol = 1e-4,
            abstol = 1e-3, maxiters = 30, batch = 10)
        test_2d_poisson_equation(chain, strategy)
    end
end

@testitem "PDE III: 3rd-order ODE" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    @parameters x
    @variables u(..), Dxu(..), Dxxu(..), O1(..), O2(..)
    Dxxx = Differential(x)^3
    Dx = Differential(x)

    # ODE
    eq = Dx(Dxxu(x)) ~ cospi(x)

    # Initial and boundary conditions
    bcs_ = [
        u(0.0) ~ 0.0,
        u(1.0) ~ cospi(1.0),
        Dxu(1.0) ~ 1.0
    ]
    ep = (cbrt(eps(eltype(Float64))))^2 / 6

    der = [
        Dxu(x) ~ Dx(u(x)) + ep * O1(x),
        Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x)
    ]

    bcs = [bcs_; der]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = [[Chain(Dense(1, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)) for _ in 1:3]
             [Chain(Dense(1, 4, tanh), Dense(4, 1)) for _ in 1:2]]
    quasirandom_strategy = QuasiRandomTraining(100; sampling_alg = LatinHypercubeSample())

    discretization = PhysicsInformedNN(chain, quasirandom_strategy)

    @named pde_system = PDESystem(eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)])

    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

    callback = function (p, l)
        if p.iter % 100 == 0 || p.iter == 1
            println("loss: ", l)
            println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
            println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
        end
        return false
    end

    res = solve(prob, BFGS(); maxiters = 1000, callback)
    phi = discretization.phi[1]

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u.depvar.u)) for x in xs]

    @test u_predict≈u_real atol=10^-4
end

@testitem "PDE IV: System of PDEs" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum

    @parameters x, y
    @variables u1(..), u2(..)
    Dx = Differential(x)
    Dy = Differential(y)

    # System of pde
    eqs = [
        Dx(u1(x, y)) + 4 * Dy(u2(x, y)) ~ 0,
        Dx(u2(x, y)) + 9 * Dy(u1(x, y)) ~ 0
    ]

    # Initial and boundary conditions
    bcs = [
        u1(x, 0) ~ 2 * x,
        u2(x, 0) ~ 3 * x
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # Neural network
    chain1 = Chain(Dense(2, 15, tanh), Dense(15, 1))
    chain2 = Chain(Dense(2, 15, tanh), Dense(15, 1))

    quadrature_strategy = QuadratureTraining(quadrature_alg = CubatureJLh(),
        reltol = 1e-3, abstol = 1e-3, maxiters = 50, batch = 100)
    chain = [chain1, chain2]

    discretization = PhysicsInformedNN(chain, quadrature_strategy)

    @named pde_system = PDESystem(eqs, bcs, domains, [x, y], [u1(x, y), u2(x, y)])

    prob = discretize(pde_system, discretization)

    res = solve(prob, Adam(0.01); maxiters = 2000, callback)
    phi = discretization.phi

    analytic_sol_func(x, y) = [1 / 3 * (6x - y), 1 / 2 * (6x - y)]
    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    u_real = [[analytic_sol_func(x, y)[i] for x in xs for y in ys] for i in 1:2]
    depvars = [:u1, :u2]

    u_predict = [[phi[i]([x, y], res.u.depvar[depvars[i]])[1] for x in xs for y in ys]
                 for i in 1:2]

    @test u_predict[1]≈u_real[1] atol=0.3 norm=Base.Fix1(maximum, abs)
    @test u_predict[2]≈u_real[2] atol=0.3 norm=Base.Fix1(maximum, abs)
end

@testitem "PDE V: 2D Wave Equation" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals,
          LineSearches, Integrals
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

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
        Dt(u(x, 0)) ~ 0.0        # for all  0 < x < 1]
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        t ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    # Neural network
    chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
    phi = NeuralPDE.Phi(chain)
    derivative = NeuralPDE.numeric_derivative

    quadrature_strategy = QuadratureTraining(quadrature_alg = CubatureJLh(),
        reltol = 1e-3, abstol = 1e-3, maxiters = 50, batch = 100)

    discretization = PhysicsInformedNN(chain, quadrature_strategy)
    prob = discretize(pde_system, discretization)

    cb_ = function (p, l)
        println("loss: ", l)
        println("losses: ", map(l -> l(p.u), loss_functions))
        return false
    end

    res = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 500, callback)

    dx = 0.1
    xs, ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    function analytic_sol_func(x, t)
        sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50000])
    end

    u_predict = reshape([first(phi([x, t], res.u)) for x in xs for t in ts],
        (length(xs), length(ts)))
    u_real = reshape([analytic_sol_func(x, t) for x in xs for t in ts],
        (length(xs), length(ts)))
    @test u_predict≈u_real atol=0.1
end

@testitem "PDE VI: PDE with mixed derivative" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
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
        u(x, 0) ~ Dy(u(x, 0))
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
    @test u_predict≈u_real rtol=0.1
end

@testitem "NNPDE: Translating from Flux" tags=[:nnpde1] setup=[NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS
    import Flux

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)
    eq = Dθ(u(θ)) ~ θ^3 + 2 * θ + (θ^2) * ((1 + 3 * (θ^2)) / (1 + θ + (θ^3))) -
                    u(θ) * (θ + ((1 + 3 * (θ^2)) / (1 + θ + θ^3)))
    bcs = [u(0.0) ~ 1.0]
    domains = [θ ∈ Interval(0.0, 1.0)]

    chain = Flux.Chain(Flux.Dense(1, 12, Flux.σ), Flux.Dense(12, 1))
    discretization = PhysicsInformedNN(chain, QuadratureTraining())
    @test discretization.chain isa Lux.AbstractLuxLayer

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(0.1); maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.01); maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.001); maxiters = 500)
    phi = discretization.phi
    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(phi(t, res.u)) for t in ts]
    @test u_predict≈u_real atol=0.1
end
