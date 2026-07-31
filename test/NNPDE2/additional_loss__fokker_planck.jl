using NeuralPDE
using Test

@testset "Fokker-Planck" begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, ComponentArrays,
        Integrals, Cubature
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS, LBFGS

    # the example took from this article https://arxiv.org/abs/1910.10503
    @parameters x
    @variables p(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    α, β, _σ = 0.3, 0.5, 0.5
    dx = 0.01

    # here we use normalization condition: dx*p(x) ~ 1, in order to get non-zero solution.
    # (α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x))
    eq = [Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))]
    x_0, x_end = -2.2, 2.2

    # Initial and boundary conditions
    bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]

    # Space and time domains
    domains = [x ∈ Interval(-2.2, 2.2)]

    # Neural network
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
        additional_loss = norm_loss_function
    )
    @named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
    prob = discretize(pde_system, discretization)

    sym_prob = symbolic_discretize(pde_system, discretization)
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    phi = discretization.phi

    callback = function (p, l)
        if p.iter % 100 == 0
            println("loss: ", l)
            println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
            println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
            println("additional_loss: ", norm_loss_function(phi, p.u, nothing))
        end
        return false
    end

    res = solve(prob, LBFGS(); maxiters = 400, callback)
    prob = remake(prob; u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 2000, callback)

    C = 142.88418699042
    analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
    xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]

    u_predict = [first(phi(x, res.u)) for x in xs]
    @test u_predict ≈ u_real rtol = 1.0e-3

    discretization = PhysicsInformedNN(
        chain, GridTraining(dx); additional_loss = norm_loss_function
    )
    @named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
    prob = discretize(pde_system, discretization)

    sym_prob = symbolic_discretize(pde_system, discretization)
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    phi = discretization.phi

    callback = function (p, l)
        if p.iter % 100 == 0
            println("loss: ", l)
            println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
            println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
            println("additional_loss: ", norm_loss_function(phi, p.u, nothing))
        end
        return false
    end

    res = solve(prob, LBFGS(); maxiters = 400, callback)
    prob = remake(prob; u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 2000, callback)

    u_predict = [first(phi(x, res.u)) for x in xs]
    @test u_predict ≈ u_real rtol = 1.0e-3
end
