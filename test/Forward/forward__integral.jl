using NeuralPDE
using Test

@testset "Integral" begin
    using DomainSets, Lux, Random, Zygote, ComponentArrays

    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(0, Inf))
    eq = I(u(x)) ~ 0
    bcs = [u(1.0) ~ exp(1) / (exp(2) + 3)]
    domains = [x ∈ Interval(1.0, 2.0)]
    chain = Chain(x -> exp.(x) ./ (exp.(2 .* x) .+ 3))
    init_params, st = Lux.setup(Random.default_rng(), chain)
    chain([1], init_params, st)
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(
        chain, strategy_;
        init_params = init_params
    )
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    sym_prob = symbolic_discretize(pde_system, discretization)
    prob = discretize(pde_system, discretization)
    inner_loss = sym_prob.loss_functions.datafree_pde_loss_functions[1]
    exact_u = π / (3 * sqrt(3))
    @test inner_loss(ones(1, 1), init_params)[1] ≈ exact_u rtol = 1.0e-5

    #infinite intervals
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(-Inf, Inf))
    eqs = I(u(x)) ~ 0
    domains = [x ∈ Interval(1.0, 2.0)]
    bcs = [u(1) ~ u(1)]
    chain = Chain(x -> x .* exp.(-x .^ 2))
    chain([1], init_params, st)

    discretization = PhysicsInformedNN(chain, strategy_; init_params)
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
    sym_prob = symbolic_discretize(pde_system, discretization)
    prob = discretize(pde_system, discretization)
    inner_loss = sym_prob.loss_functions.datafree_pde_loss_functions[1]
    exact_u = 0
    @test inner_loss(ones(1, 1), init_params)[1] ≈ exact_u atol = 1.0e-13
end
