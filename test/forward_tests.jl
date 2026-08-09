@testitem "Forward ODE Evaluation" tags = [:pinnparser] begin
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
    discretization = PhysicsInformedNN(chain, strategy_; init_params)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    sym_prob = symbolic_discretize(pde_system, discretization)
    inner_loss = sym_prob.loss_functions.datafree_pde_loss_functions[1]
    xs = rand(1, 10)
    dudx(x) = @. 2 * x
    @test inner_loss(xs, init_params) ≈ dudx(xs) rtol = 1.0e-8
end

@testitem "Forward Derivatives Evaluation" tags = [:pinnparser] begin
    using DomainSets, Lux, Random, Zygote, ComponentArrays, NeuralPDE
    import DomainSets: Interval

    @parameters x
    @variables u(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    eq = Dxx(u(x)) - Dx(u(x)) ~ 0
    bcs = [u(1.0) ~ exp(1)]
    domains = [x ∈ Interval(1.0, 2.0)]

    chain = Chain(x -> exp.(x))
    init_params, st = Lux.setup(Random.default_rng(), chain)
    init_params = init_params |> ComponentArray{Float64}

    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_; init_params)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    sym_prob = symbolic_discretize(pde_system, discretization)
    prob = discretize(pde_system, discretization)
    inner_loss = sym_prob.loss_functions.datafree_pde_loss_functions[1]

    @test inner_loss(ones(1, 1), init_params)[1] ≈ 0.0 atol = 1.0e-5
    @test prob.f(init_params, nothing) ≈ 0.0 atol = 1.0e-5
end
