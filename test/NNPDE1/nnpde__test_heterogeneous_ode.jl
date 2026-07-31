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

@testset "Test Heterogeneous ODE" begin
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
        @test u_predict ≈ u_real atol = 0.8
    end

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        simple_1d_ode(strategy)
    end
end
