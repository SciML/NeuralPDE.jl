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

@testset "PDE III: 3rd-order ODE" begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals,
        ComponentArrays
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
    # BFGS requires the objective to stay fixed between evaluations, and Sobol sampling
    # keeps the collocation set independent of the global RNG.
    quasirandom_strategy = QuasiRandomTraining(
        100; sampling_alg = SobolSample(), resampling = false, minibatch = 1
    )

    init_params = let rng = Xoshiro(100)
        [ComponentArray{Float64}(Lux.initialparameters(rng, c)) for c in chain]
    end

    discretization = PhysicsInformedNN(chain, quasirandom_strategy; init_params)

    @named pde_system = PDESystem(
        eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)]
    )

    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

    # Stop on the loss level the accuracy assertion needs instead of on a fixed iteration
    # count. BFGS is still descending at 1000 iterations here, so a truncated run lands
    # wherever floating-point differences (code coverage, BLAS, hardware) put it.
    training_tol = 1.0e-9

    callback = function (p, l)
        if p.iter % 100 == 0||p.iter == 1
            println("loss: ", l)
            println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
            println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
        end
        return l < training_tol
    end

    res = solve(prob, BFGS(); maxiters = 5000, callback)
    phi = discretization.phi[1]

    @test res.objective < training_tol

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u.depvar.u)) for x in xs]

    @test u_predict ≈ u_real atol = 10^-4
end
