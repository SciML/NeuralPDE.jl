@testsetup module SymbolicNNPDE1TestSetup

using NeuralPDE, Cubature, Integrals, QuasiMonteCarlo

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

@testitem "Symbolic PINN parser: Test Heterogeneous ODE" tags = [:symbolicpinn] setup = [SymbolicNNPDE1TestSetup] begin
    using Cubature, Integrals, QuasiMonteCarlo, DomainSets, Lux, Random, Optimisers
    import DomainSets: Interval, infimum, supremum

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

        discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
        @named pde_system = PDESystem(eq, bcs, domains, [θ], [u(θ)])
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
        @test u_predict ≈ u_real atol = 0.8
    end

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        simple_1d_ode(strategy)
    end
end



@testitem "Symbolic PINN parser: PDE II - 2D Poisson" tags = [:symbolicpinn] setup = [SymbolicNNPDE1TestSetup] begin
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

        discretization = PhysicsInformedNN(chain, strategy; init_params = ps, symbolic_parser = true)
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

@testitem "Symbolic PINN parser: PDE III - 3rd-order ODE" tags = [:symbolicpinn] setup = [SymbolicNNPDE1TestSetup] begin
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
        Dxu(1.0) ~ 1.0,
    ]
    ep = (cbrt(eps(eltype(Float64))))^2 / 6

    der = [
        Dxu(x) ~ Dx(u(x)) + ep * O1(x),
        Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x),
    ]

    bcs = [bcs_; der]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = [
        [Chain(Dense(1, 12, tanh), Dense(12, 12, tanh), Dense(12, 1)) for _ in 1:3]
        [Chain(Dense(1, 4, tanh), Dense(4, 1)) for _ in 1:2]
    ]
    quasirandom_strategy = QuasiRandomTraining(100; sampling_alg = LatinHypercubeSample())

    discretization = PhysicsInformedNN(chain, quasirandom_strategy; symbolic_parser = true)

    @named pde_system = PDESystem(
        eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)]
    )

    prob = discretize(pde_system, discretization)

    res = solve(prob, BFGS(); maxiters = 1000, callback)
    phi = discretization.phi[1]

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u.depvar.u)) for x in xs]

    @test u_predict ≈ u_real atol = 10^-4
end

@testitem "Symbolic PINN parser: PDE IV - System of PDEs" tags = [:symbolicpinn] setup = [SymbolicNNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum

    @parameters x, y
    @variables u1(..), u2(..)
    Dx = Differential(x)
    Dy = Differential(y)

    # System of pde
    eqs = [
        Dx(u1(x, y)) + 4 * Dy(u2(x, y)) ~ 0,
        Dx(u2(x, y)) + 9 * Dy(u1(x, y)) ~ 0,
    ]

    # Initial and boundary conditions
    bcs = [
        u1(x, 0) ~ 2 * x,
        u2(x, 0) ~ 3 * x,
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # Neural network
    chain1 = Chain(Dense(2, 15, tanh), Dense(15, 1))
    chain2 = Chain(Dense(2, 15, tanh), Dense(15, 1))

    quadrature_strategy = QuadratureTraining(
        quadrature_alg = CubatureJLh(),
        reltol = 1.0e-3, abstol = 1.0e-3, maxiters = 50, batch = 100
    )
    chain = [chain1, chain2]

    discretization = PhysicsInformedNN(chain, quadrature_strategy; symbolic_parser = true)

    @named pde_system = PDESystem(eqs, bcs, domains, [x, y], [u1(x, y), u2(x, y)])

    prob = discretize(pde_system, discretization)

    res = solve(prob, Adam(0.01); maxiters = 2000, callback)
    phi = discretization.phi

    analytic_sol_func(x, y) = [1 / 3 * (6x - y), 1 / 2 * (6x - y)]
    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    u_real = [[analytic_sol_func(x, y)[i] for x in xs for y in ys] for i in 1:2]
    depvars = [:u1, :u2]

    u_predict = [
        [phi[i]([x, y], res.u.depvar[depvars[i]])[1] for x in xs for y in ys]
            for i in 1:2
    ]

    @test u_predict[1] ≈ u_real[1] atol = 0.3 norm = Base.Fix1(maximum, abs)
    @test u_predict[2] ≈ u_real[2] atol = 0.3 norm = Base.Fix1(maximum, abs)
end



@testsetup module SymbolicAdaptiveLossTestSetup
using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, NeuralPDE, Test,
    TensorBoardLogger
import DomainSets: Interval, infimum, supremum

function solve_with_adaptive_loss(
        adaptive_loss; haslogger = false, outdir = mktempdir(), run = 1
    )
    logdir = joinpath(outdir, string(run))
    logger = haslogger ? TBLogger(logdir) : nothing

    Random.seed!(60)
    hid = 40
    chain = Chain(Dense(2, hid, tanh), Dense(hid, hid, tanh), Dense(hid, 1))
    strategy = StochasticTraining(256)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ -sinpi(1) * sinpi(y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sinpi(x) * sinpi(1),
    ]

    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    discretization = PhysicsInformedNN(chain, strategy; adaptive_loss, logger, symbolic_parser = true)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)
    u_real = [analytic_sol_func(x, y) for x in xs for y in ys]

    callback = function (p, l)
        if p.iter % 250 == 0
            @info "[$(nameof(typeof(adaptive_loss)))] Current loss is: $l, iteration is $(p.iter)"
        end
        return false
    end

    res = solve(prob, Adam(0.03); maxiters = 2500, callback)
    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]

    total_diff = sum(abs, u_predict .- u_real)
    total_u = sum(abs, u_real)
    total_diff_rel = total_diff / total_u

    return total_diff_rel
end

export solve_with_adaptive_loss

end

@testitem "Symbolic PINN parser: 2D Poisson NonAdaptiveLoss" tags = [:symbolicpinn] setup = [SymbolicAdaptiveLossTestSetup] begin
    loss = NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)
    tmpdir = mktempdir()
    total_diff_rel = solve_with_adaptive_loss(loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
end



@testitem "Symbolic PINN parser: 2D Poisson MiniMaxAdaptiveLoss" tags = [:symbolicpinn] setup = [SymbolicAdaptiveLossTestSetup] begin
    loss = MiniMaxAdaptiveLoss(100; pde_loss_weights = 1, bc_loss_weights = 1)
    tmpdir = mktempdir()
    total_diff_rel = solve_with_adaptive_loss(loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
end

@testitem "Symbolic PINN parser: Fokker-Planck" tags = [:symbolicpinn] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, ComponentArrays,
        Integrals, Cubature
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS, LBFGS

    @parameters x
    @variables p(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    α, β, _σ = 0.3, 0.5, 0.5
    dx = 0.01

    eq = [Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))]
    x_0, x_end = -2.2, 2.2

    bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]
    domains = [x ∈ Interval(-2.2, 2.2)]

    inn = 18
    chain = Chain(Dense(1, inn, σ), Dense(inn, inn, σ), Dense(inn, inn, σ), Dense(inn, 1))

    init_params = ComponentArray{Float64}(
        Lux.initialparameters(
            Random.default_rng(), chain
        )
    )

    lb, ub = [x_0], [x_end]

    function norm_loss_function(phi, θ, p)
        inner_f(x, θ) = dx * phi(x, θ) .- 1
        prob1 = IntegralProblem(inner_f, (lb, ub), θ)
        norm2 = solve(prob1, HCubatureJL(), reltol = 1.0e-8, abstol = 1.0e-8, maxiters = 10)
        return abs(norm2[1])
    end

    discretization = PhysicsInformedNN(
        chain, GridTraining(dx); init_params,
        additional_loss = norm_loss_function,
        symbolic_parser = true
    )
    @named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
    prob = discretize(pde_system, discretization)
    phi = discretization.phi

    res = solve(prob, LBFGS(); maxiters = 400)
    prob = remake(prob; u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 2000)

    C = 142.88418699042
    analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
    xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]

    u_predict = [first(phi(x, res.u)) for x in xs]
    @test u_predict ≈ u_real rtol = 1.0e-3
end

@testitem "Symbolic PINN parser: Approximation from data and additional_loss" tags = [:symbolicpinn] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Optimisers,
        ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Lux
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    @parameters x
    @variables u(..)

    eq = [u(0) ~ u(0)]
    bc = [u(0) ~ u(0)]
    x0 = 0
    x_end = pi
    dx = pi / 10

    domain = [x ∈ Interval(x0, x_end)]
    hidden = 10

    chain = Chain(
        Dense(1, hidden, tanh), Dense(hidden, hidden, sin),
        Dense(hidden, hidden, tanh), Dense(hidden, 1)
    )

    strategy = GridTraining(dx)
    xs = collect(x0:dx:x_end)'

    aproxf(x) = @. cospi(x)
    data = aproxf(xs)

    additional_loss(phi, θ, p) = sum(abs2, phi(xs, θ) .- data)

    discretization = PhysicsInformedNN(chain, strategy; additional_loss, symbolic_parser = true)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)

    res = solve(prob, Adam(0.01); maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 500)
    phi = discretization.phi

    @test phi(xs, res.u) ≈ aproxf(xs) rtol = 0.02
end

@testitem "Symbolic PINN parser: Direct Function Approximation 1D" tags = [:symbolicpinn] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x
    @variables u(..)

    func(x) = @. 2 + abs(x - 0.5)

    eq = [u(x) ~ func(x)]
    bc = [u(0) ~ u(0)]

    x0 = 0
    x_end = 2
    dx = 0.001
    domain = [x ∈ Interval(x0, x_end)]

    xs = collect(x0:dx:x_end)

    hidden = 10
    chain = Chain(Dense(1, hidden, tanh), Dense(hidden, hidden, tanh), Dense(hidden, 1))

    strategy = GridTraining(0.01)
    discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(0.05), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(initial_stepnorm = 0.01), maxiters = 500)

    @test discretization.phi(xs', res.u) ≈ func(xs') rtol = 0.02
end

@testitem "Symbolic PINN parser: Direct Function Approximation 2D" tags = [:symbolicpinn] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x, y
    @variables u(..)
    func(x, y) = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))

    eq = [u(x, y) ~ func(x, y)]
    bc = [u(0, 0) ~ u(0, 0)]

    x0 = -10
    x_end = 10
    y0 = -10
    y_end = 10
    d = 0.4
    domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]
    hidden = 25
    chain = Chain(
        Dense(2, hidden, tanh), Dense(hidden, hidden, tanh),
        Dense(hidden, hidden, tanh), Dense(hidden, 1)
    )
    strategy = GridTraining(d)
    discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
    @named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 500)
    phi = discretization.phi
    xs = collect(x0:0.1:x_end)
    ys = collect(y0:0.1:y_end)
    u_predict = reshape(
        [first(phi([x, y], res.u)) for x in xs for y in ys],
        (length(xs), length(ys))
    )
    u_real = reshape([func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
    @test u_predict ≈ u_real rtol = 0.05
end

@testitem "Symbolic PINN parser: Trivial BC [0 ~ 0] fails for some training strategies" tags = [:symbolicpinn] begin
    using NeuralPDE, Optimization, OptimizationOptimisers, Lux, DomainSets
    @parameters x
    @variables u(..)

    eq = [u(x) ~ 2 + abs(x - 0.5)]
    bc = [0 ~ 0]
    domain = [x ∈ Interval(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    for strategy in (StochasticTraining(1000), QuasiRandomTraining(1000))
        discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        @test_throws ArgumentError discretize(pde_system, discretization)
    end
end

@testitem "Symbolic PINN parser: Empty boundary condition [] fails in solve phase" tags = [:symbolicpinn] begin
    using NeuralPDE, Optimization, OptimizationOptimisers, Lux, DomainSets
    @parameters x
    @variables u(..)

    eq = [u(x) ~ 2 + abs(x - 0.5)]
    bc = []
    domain = [x ∈ Interval(0.0, 2.0)]
    chain = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

    for strategy in (
            GridTraining(0.01),
            StochasticTraining(1000),
            QuasiRandomTraining(1000),
            QuadratureTraining(),
        )
        discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
        @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
        prob = discretize(pde_system, discretization)
        @test_throws MethodError solve(prob, Adam(0.05), maxiters = 10)
    end
end

# ==============================================================================
# INTEGRO-DIFFERENTIAL EQUATIONS (IDE) TESTS WITH SYMBOLIC PARSER
# ==============================================================================

@testsetup module SymbolicIDETestSetup
function callback(p, l)
    if p.iter == 1 || p.iter % 10 == 0
        println("Current loss is: $l after $(p.iter) iterations")
    end
    return false
end
export callback
end

@testitem "Symbolic PINN parser: IntegroDiff Example 1 -- 1D" tags = [:symbolicpinn] setup = [SymbolicIDETestSetup] begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics, NeuralPDE
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters t
    @variables i(..)
    Di = Differential(t)
    Ii = Integral(t in DomainSets.ClosedInterval(0, t))
    eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(t)) ~ 1
    bcs = [i(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chain = Chain(Dense(1, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
    @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    analytic_sol_func(t) = 1 / 2 * (exp(-t)) * (sin(2 * t))

    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(phi([t], res.u)) for t in ts]
    @test mean(abs2, u_real .- u_predict) < 0.02
end



@testitem "Symbolic PINN parser: IntegroDiff Example 3 -- 2 Inputs, 1 Output" tags = [:symbolicpinn] setup = [SymbolicIDETestSetup] begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics, NeuralPDE
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x, y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Ix = Integral((x, y) in DomainSets.UnitSquare())

    eq = Ix(u(x, y)) ~ 1 / 3
    bcs = [u(0.0, 0.0) ~ 1, Dx(u(x, y)) ~ -2 * x, Dy(u(x, y)) ~ -2 * y]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(2, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy; symbolic_parser = true)
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    phi = discretization.phi

    xs = 0.0:0.01:1.0
    ys = 0.0:0.01:1.0

    u_real = collect(1 - x^2 - y^2 for y in ys, x in xs)
    u_predict = collect(Array(phi([x, y], res.u))[1] for y in ys, x in xs)
    @test mean(abs2, u_real .- u_predict) < 0.001
end

# ==============================================================================
# BAYESIAN PINN (BPINN) TESTS WITH SYMBOLIC PARSER
# ==============================================================================

@testitem "Symbolic PINN parser: BPINN PDE I - 1D Periodic System" tags = [:symbolicpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t
    @variables u(..)
    Dt = Differential(t)
    eq = Dt(u(t)) - cospi(2t) ~ 0
    bcs = [u(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chainl = Chain(Dense(1, 6, tanh), Dense(6, 1))
    @named pde_system = PDESystem(eq, bcs, domains, [t], [u(t)])

    discretization = BayesianPINN([chainl], GridTraining([0.01]); symbolic_parser = true)

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 1500, bcstd = [0.01],
        phystd = [0.01], priorsNNw = (0.0, 1.0), saveats = [1 / 50.0]
    )

    analytic_sol_func(u0, t) = u0 + sinpi(2t) / (2pi)
    ts = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(0.0, t) for t in ts]
    u_predict = pmean(sol1.ensemblesol[1])

    @test mean(abs, u_predict .- u_real) < 8.0e-2
end

@testitem "Symbolic PINN parser: BPINN PDE II - 1D ODE" tags = [:symbolicpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    eq = Dθ(u(θ)) ~
        θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

    bcs = [u(0.0) ~ 1.0f0]
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    chain = Chain(Dense(1, 12, σ), Dense(12, 1))
    discretization = BayesianPINN([chain], GridTraining([0.01]); symbolic_parser = true)
    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 500, bcstd = [0.1],
        phystd = [0.05], priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol1.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol1.ensemblesol[1])
    @test u_predict ≈ u_real atol = 0.8
end

@testitem "Symbolic PINN parser: BPINN PDE III - 2D Poisson" tags = [:symbolicpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0,
        u(x, 1) ~ 0.0,
    ]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(2, 9, σ), Dense(9, 9, σ), Dense(9, 1))
    dx = 0.04
    discretization = BayesianPINN([chain], GridTraining(dx); symbolic_parser = true)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    sol = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 200,
        bcstd = [0.003, 0.003, 0.003, 0.003], phystd = [0.003],
        priorsNNw = (0.0, 10.0), saveats = [1 / 100.0, 1 / 100.0]
    )

    xs = sol.timepoints[1]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)

    u_predict = pmean(sol.ensemblesol[1])
    u_real = [analytic_sol_func(xs[:, i][1], xs[:, i][2]) for i in 1:length(xs[1, :])]
    @test u_predict ≈ u_real rtol = 0.5
end

# ==============================================================================
# FORWARD EVALUATION TESTS WITH SYMBOLIC PARSER (from forward_tests.jl)
# ==============================================================================

@testitem "Symbolic PINN parser: Forward ODE Evaluation" tags = [:symbolicpinn] begin
    using DomainSets, Lux, Random, Zygote, ComponentArrays, Adapt, NeuralPDE
    import DomainSets: Interval

    @parameters x
    @variables u(..)

    Dx = Differential(x)
    eq = Dx(u(x)) ~ 0.0
    bcs = [u(0.0) ~ u(0.0)]
    domains = [x ∈ Interval(0.0, 1.0)]
    chain = Chain(x -> x .^ 2)
    init_params, st = Lux.setup(Random.default_rng(), chain)
    init_params = init_params |> ComponentArray{Float64}

    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_; init_params, symbolic_parser = true)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    sym_prob = symbolic_discretize(pde_system, discretization)
    prob = discretize(pde_system, discretization)

    eqs = pde_system.eqs
    bcs = pde_system.bcs
    domains = pde_system.domain
    dx = strategy_.dx
    eltypeθ = eltype(discretization.init_params)
    depvars, indvars, dict_indvars,
        dict_depvars, dict_depvar_input = NeuralPDE.get_vars(
        pde_system.ivs, pde_system.dvs
    )

    train_sets = generate_training_sets(
        domains, dx, eqs, bcs, eltypeθ,
        dict_indvars, dict_depvars
    )

    pde_train_sets, bcs_train_sets = train_sets |> NeuralPDE.EltypeAdaptor{eltypeθ}()
    pde_train_sets = first(pde_train_sets)

    train_data = pde_train_sets
    pde_loss_function = sym_prob.loss_functions.datafree_pde_loss_functions[1]

    dudx(x) = @. 2 * x
    @test pde_loss_function(train_data, init_params) ≈ dudx(train_data) rtol = 1.0e-8
end

@testitem "Symbolic PINN parser: Forward Derivatives Evaluation" tags = [:symbolicpinn] begin
    using DomainSets, Lux, Random, Zygote, ComponentArrays, NeuralPDE

    chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
    init_params = Lux.initialparameters(Random.default_rng(), chain) |>
        ComponentArray{Float64}

    phi = NeuralPDE.Phi(chain)
    derivative = NeuralPDE.numeric_derivative

    u_ = (cord, θ, phi) -> sum(phi(cord, θ))

    phi_ = (p) -> phi(p, init_params)[1]
    dphi = Zygote.gradient(phi_, [1.0, 2.0])

    eps_x = NeuralPDE.get_ε(2, 1, Float64, 1)
    eps_y = NeuralPDE.get_ε(2, 2, Float64, 1)

    dphi_x = derivative(phi, u_, [1.0, 2.0], [eps_x], 1, init_params)
    dphi_y = derivative(phi, u_, [1.0, 2.0], [eps_y], 1, init_params)

    # first order derivatives
    @test isapprox(dphi[1][1], dphi_x, atol = 1.0e-8)
    @test isapprox(dphi[1][2], dphi_y, atol = 1.0e-8)

    eps_x = NeuralPDE.get_ε(2, 1, Float64, 2)
    eps_y = NeuralPDE.get_ε(2, 2, Float64, 2)

    hess_phi = Zygote.hessian(phi_, [1, 2])

    dphi_xx = derivative(phi, u_, [1.0, 2.0], [eps_x, eps_x], 2, init_params)
    dphi_xy = derivative(phi, u_, [1.0, 2.0], [eps_x, eps_y], 2, init_params)
    dphi_yy = derivative(phi, u_, [1.0, 2.0], [eps_y, eps_y], 2, init_params)

    # second order derivatives
    @test isapprox(hess_phi[1], dphi_xx, atol = 4.0e-5)
    @test isapprox(hess_phi[2], dphi_xy, atol = 4.0e-5)
    @test isapprox(hess_phi[4], dphi_yy, atol = 4.0e-5)
end





