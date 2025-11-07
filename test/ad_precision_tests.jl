using Test
using NeuralPDE, Lux, Random, ComponentArrays
using Optimization
using OptimizationOptimisers
using DomainSets: Interval
using ModelingToolkit: @parameters, @variables, PDESystem, Differential
using Printf

# Test for issue #931: Precision loss in AutoDiff through the loss function
@testset "AD Precision Tests (Issue #931)" begin
    using ForwardDiff, DifferentiationInterface, LinearAlgebra

    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)
    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0

    # 2D PDE
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
    # Initial and boundary conditions
    bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max)]

    # Neural network
    inner = 25
    chain = Chain(Dense(3, inner, σ), Dense(inner, 1))

    strategy = GridTraining(0.1)
    ps, st = Lux.setup(Random.default_rng(), chain)
    ps = ps |> ComponentArray .|> Float64
    discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

    @named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    prob = discretize(pde_system, discretization)
    symprob = symbolic_discretize(pde_system, discretization)

    # Get the full residual function
    function get_residual_vector(pinnrep, loss_function, train_set)
        eltypeθ = NeuralPDE.recursive_eltype(pinnrep.flat_init_params)
        train_set = train_set |> NeuralPDE.safe_get_device(pinnrep.init_params) |>
                    NeuralPDE.EltypeAdaptor{eltypeθ}()
        return θ -> loss_function(train_set, θ)
    end

    function get_full_residual(prob, symprob)
        # Get training sets
        (; domains, eqs, bcs, dict_indvars, dict_depvars, strategy) = symprob
        eltypeθ = NeuralPDE.recursive_eltype(symprob.flat_init_params)
        adaptor = NeuralPDE.EltypeAdaptor{eltypeθ}()

        train_sets = NeuralPDE.generate_training_sets(domains, strategy.dx, eqs, bcs,
            eltypeθ,
            dict_indvars, dict_depvars)
        pde_train_sets, bcs_train_sets = train_sets |> adaptor

        # Get residuals
        pde_residuals = [get_residual_vector(symprob, _loss, _set)
                         for (_loss, _set) in zip(
            symprob.loss_functions.datafree_pde_loss_functions, pde_train_sets)]
        bc_residuals = [get_residual_vector(symprob, _loss, _set)
                        for (_loss, _set) in zip(
            symprob.loss_functions.datafree_bc_loss_functions, bcs_train_sets)]

        # Setup adaloss weights (assuming NonAdaptiveLoss)
        num_pde_losses = length(pde_residuals)
        num_bc_losses = length(bc_residuals)
        adaloss = symprob.adaloss
        adaloss_T = eltype(adaloss.pde_loss_weights)

        function full_residual(θ)
            pde_losses = [pde_residual(θ) for pde_residual in pde_residuals]
            bc_losses = [bc_residual(θ) for bc_residual in bc_residuals]

            weighted_pde_losses = sqrt.(adaloss.pde_loss_weights) .* pde_losses ./
                                  sqrt.(length.(pde_losses))
            weighted_bc_losses = sqrt.(adaloss.bc_loss_weights) .* bc_losses ./
                                 sqrt.(length.(bc_losses))

            full_res = hcat(hcat(weighted_pde_losses...), hcat(weighted_bc_losses...))
            return full_res
        end

        return full_residual
    end

    residual = get_full_residual(prob, symprob)
    loss = θ -> sum(abs2, residual(θ))
    loss_neuralpdes = θ -> prob.f(θ, prob.p)

    θ = prob.u0

    # Test 1: Sanity check that our loss matches NeuralPDE's loss
    rel_err = abs(loss_neuralpdes(θ) - loss(θ)) / abs(loss_neuralpdes(θ))
    @test rel_err < 1e-14
    println("Loss function match error: $rel_err")

    # Test 2: Check JVP precision on the residual function
    v = randn(length(θ))
    J_fwd = ForwardDiff.jacobian(residual, θ)
    jvp_explicit = J_fwd * v
    jvp_pushforward = DifferentiationInterface.pushforward(
        residual,
        AutoForwardDiff(),
        θ,
        (v,),
    )[1]

    jvp_error = norm(jvp_explicit - jvp_pushforward[:]) / norm(jvp_explicit)
    println("AutoForwardDiff error on residual jvp: $jvp_error")

    # This is the key test: the JVP error should be at Float64 precision (< 1e-14)
    # Previously this would be ~1e-8 due to precision loss
    @test jvp_error < 1e-12

    # Test 3: Verify model evaluation also maintains precision
    function get_quadpoints(symprob, strategy)
        (; domains, eqs, dict_indvars, dict_depvars) = symprob
        eltypeθ = NeuralPDE.recursive_eltype(symprob.flat_init_params)

        train_sets = hcat(NeuralPDE.generate_training_sets(domains, strategy.dx, eqs, [],
            eltypeθ,
            dict_indvars, dict_depvars)[1]...)
        return train_sets
    end

    x_points = get_quadpoints(symprob, strategy)
    fun = ps -> chain(x_points, ps, st)[1]
    J_fwd_model = ForwardDiff.jacobian(fun, θ)
    jvp_explicit_model = J_fwd_model * v
    jvp_pushforward_model = DifferentiationInterface.pushforward(
        fun,
        AutoForwardDiff(),
        θ,
        (v,),
    )[1]

    model_jvp_error = norm(jvp_explicit_model - jvp_pushforward_model[:]) /
                      norm(jvp_explicit_model)
    println("AutoForwardDiff error on model jvp: $model_jvp_error")
    @test model_jvp_error < 1e-14
end
