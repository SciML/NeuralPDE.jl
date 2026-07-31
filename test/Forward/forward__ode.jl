using NeuralPDE
using Test

@testset "ODE" begin
    using DomainSets, Lux, Random, Zygote, ComponentArrays, Adapt

    @parameters x
    @variables u(..)

    Dx = Differential(x)
    eq = Dx(u(x)) ~ 0.0
    bcs = [u(0.0) ~ u(0.0)]
    domains = [x ∈ Interval(0.0, 1.0)]
    chain = Chain(x -> x .^ 2)
    init_params, st = Lux.setup(Random.default_rng(), chain)
    init_params = init_params |> ComponentArray{Float64}

    chain([1], init_params, st)
    strategy_ = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy_; init_params)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    eqs = pde_system.eqs
    bcs = pde_system.bcs
    domains = pde_system.domain
    dx = strategy_.dx
    eltypeθ = eltype(sym_prob.flat_init_params)
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
