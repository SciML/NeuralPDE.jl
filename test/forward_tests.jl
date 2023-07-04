using Test, NeuralPDE
using SciMLBase
using DomainSets
import ModelingToolkit: Interval
import Lux, Random, Zygote, Flux
using ComponentArrays

@testset "ODE" begin
    @parameters x
    @variables u(..)

    Dx = Differential(x)
    eq = Dx(u(x)) ~ 0.0
    bcs = [u(0.0) ~ u(0.0)]
    domains = [x ∈ Interval(0.0, 1.0)]
    chain = Lux.Chain(x -> x .^ 2)
    init_params, st = Lux.setup(Random.default_rng(), chain)
    init_params = Float64[]

    chain([1], Float64[], st)
    strategy_ = NeuralPDE.GridTraining(0.1)
    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy_; init_params = Float64[])
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = NeuralPDE.discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

    eqs = pde_system.eqs
    bcs = pde_system.bcs
    domains = pde_system.domain
    dx = strategy_.dx
    eltypeθ = eltype(sym_prob.flat_init_params)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(pde_system.ivs,
                                                                                         pde_system.dvs)

    train_sets = generate_training_sets(domains, dx, eqs, bcs, eltypeθ,
                                        dict_indvars, dict_depvars)

    pde_train_sets, bcs_train_sets = train_sets
    pde_train_sets = NeuralPDE.adapt(eltypeθ, pde_train_sets)[1]

    train_data = pde_train_sets
    pde_loss_function = sym_prob.loss_functions.datafree_pde_loss_functions[1]

    dudx(x) = @. 2 * x
    @test pde_loss_function(train_data, Float64[])≈dudx(train_data) rtol=1e-8
end

@testset "derivatives" begin
    chain = Flux.Chain(Flux.Dense(2, 16, Lux.σ), Flux.Dense(16, 16, Lux.σ),
                       Flux.Dense(16, 1)) |> Flux.f64
    init_params = Flux.destructure(chain)[1]

    eltypeθ = eltype(init_params)
    phi = NeuralPDE.Phi(chain)
    derivative = NeuralPDE.numeric_derivative

    u_ = (cord, θ, phi) -> sum(phi(cord, θ))

    phi([1, 2], init_params)

    phi_ = (p) -> phi(p, init_params)[1]
    dphi = Zygote.gradient(phi_, [1.0, 2.0])

    eps_x = NeuralPDE.get_ε(2, 1, Float64, 1)
    eps_y = NeuralPDE.get_ε(2, 2, Float64, 1)

    dphi_x = derivative(phi, u_, [1.0, 2.0], [eps_x], 1, init_params)
    dphi_y = derivative(phi, u_, [1.0, 2.0], [eps_y], 1, init_params)

    #first order derivatives
    @test isapprox(dphi[1][1], dphi_x, atol = 1e-8)
    @test isapprox(dphi[1][2], dphi_y, atol = 1e-8)

    eps_x = NeuralPDE.get_ε(2, 1, Float64, 2)
    eps_y = NeuralPDE.get_ε(2, 2, Float64, 2)

    hess_phi = Zygote.hessian(phi_, [1, 2])

    dphi_xx = derivative(phi, u_, [1.0, 2.0], [eps_x, eps_x], 2, init_params)
    dphi_xy = derivative(phi, u_, [1.0, 2.0], [eps_x, eps_y], 2, init_params)
    dphi_yy = derivative(phi, u_, [1.0, 2.0], [eps_y, eps_y], 2, init_params)

    #second order derivatives
    @test isapprox(hess_phi[1], dphi_xx, atol = 4e-5)
    @test isapprox(hess_phi[2], dphi_xy, atol = 4e-5)
    @test isapprox(hess_phi[4], dphi_yy, atol = 4e-5)
end

@testset "Integral" begin
    #semi-infinite intervals
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(0, Inf))
    eq = I(u(x)) ~ 0
    bcs = [u(1.0) ~ exp(1) / (exp(2) + 3)]
    domains = [x ∈ Interval(1.0, 2.0)]
    chain = Lux.Chain(x -> exp.(x) ./ (exp.(2 .* x) .+ 3))
    init_params, st = Lux.setup(Random.default_rng(), chain)
    init_params = Float64[]

    chain([1], init_params, st)
    strategy_ = NeuralPDE.GridTraining(0.1)
    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy_;
                                                 init_params = init_params)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    sym_prob = SciMLBase.symbolic_discretize(pde_system, discretization)
    prob = NeuralPDE.discretize(pde_system, discretization)
    inner_loss = sym_prob.loss_functions.datafree_pde_loss_functions[1]
    exact_u = π / (3 * sqrt(3))
    @test inner_loss(ones(1, 1), init_params)[1]≈exact_u rtol=1e-5

    #infinite intervals
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(-Inf, Inf))
    eqs = I(u(x)) ~ 0
    domains = [x ∈ Interval(1.0, 2.0)]
    bcs = [u(1) ~ u(1)]
    chain = Lux.Chain(x -> x .* exp.(-x .^ 2))
    chain([1], init_params, st)

    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy_;
                                                 init_params = init_params)
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
    sym_prob = SciMLBase.symbolic_discretize(pde_system, discretization)
    prob = SciMLBase.discretize(pde_system, discretization)
    inner_loss = sym_prob.loss_functions.datafree_pde_loss_functions[1]
    exact_u = 0
    @test inner_loss(ones(1, 1), init_params)[1]≈exact_u rtol=1e-9
end
