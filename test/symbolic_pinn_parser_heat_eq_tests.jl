@testitem "Symbolic PINN parser heat equation MVP" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux
    using Test
    import DomainSets: Interval

    function heat_equation_system()
        @parameters x t
        @variables u(..)
        Dt = Differential(t)
        Dxx = Differential(x)^2

        eq = Dt(u(x, t)) ~ Dxx(u(x, t))
        bcs = [
            u(0.0, t) ~ 0.0,
            u(1.0, t) ~ 0.0,
            u(x, 0.0) ~ sin(pi * x),
        ]
        domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
        @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])
        return heat_sys
    end

    scalar_value(x::Number) = x
    scalar_value(x) = first(x)

    heat_sys = heat_equation_system()
    parsed = NeuralPDE.parse_pde_system(heat_sys)

    @test length(parsed.eqs) == 1
    @test length(parsed.bcs) == 3
    @test length(parsed.ivs) == 2
    @test length(parsed.dvs) == 1

    chain = Lux.Chain(
        Lux.Dense(2, 8, tanh),
        Lux.Dense(8, 1)
    )

    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
        heat_sys, chain; n_interior = 3, n_bc = 3
    )

    @test length(symbolic_loss.pde_residuals) == 1
    @test length(symbolic_loss.bc_residuals) == 3

    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.pde_residuals
    )
    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.bc_residuals
    )

    theta0 = symbolic_loss.theta0
    @test !isempty(theta0)
    @test all(isfinite, theta0)

    pde_val = symbolic_loss.residual_functions.pde[1]([0.25, 0.5], theta0)
    @test isfinite(scalar_value(pde_val))

    bc_vals = [f([0.25, 0.5], theta0) for f in symbolic_loss.residual_functions.bc]
    @test all(val -> isfinite(scalar_value(val)), bc_vals)

    pde_loss = symbolic_loss.pde_loss(theta0)
    bc_loss = symbolic_loss.bc_loss(theta0)
    full_loss = symbolic_loss.loss(theta0)

    @test isfinite(pde_loss)
    @test isfinite(bc_loss)
    @test isfinite(full_loss)
    @test pde_loss >= 0
    @test bc_loss >= 0
    @test full_loss >= 0
    @test full_loss == pde_loss + bc_loss

    # Verify collocation points are stored as (D, N) matrices
    @test symbolic_loss.points.pde isa Matrix{Float64}
    @test symbolic_loss.points.bc isa Matrix{Float64}
    @test size(symbolic_loss.points.pde, 1) == 2  # D = 2 (x, t)
    @test size(symbolic_loss.points.bc, 1) == 2

    # Verify batched loss matches manual point-by-point evaluation
    pde_fn = symbolic_loss.datafree_pde_loss_functions[1]
    pde_pts = symbolic_loss.points.pde
    manual_loss = sum(abs2, pde_fn(pde_pts, theta0)) / size(pde_pts, 2)
    @test pde_loss ≈ manual_loss
end

@testitem "Symbolic PINN parser datafree loss function format" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Statistics
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
        heat_sys, chain; n_interior = 3, n_bc = 3
    )

    theta0 = symbolic_loss.theta0

    # Verify that datafree functions return the (1, N) matrix format
    # expected by NeuralPDE training strategies
    test_cord = rand(2, 5)
    for f in symbolic_loss.datafree_pde_loss_functions
        result = f(test_cord, theta0)
        @test size(result) == (1, 5)
        @test all(isfinite, result)
    end
    for f in symbolic_loss.datafree_bc_loss_functions
        result = f(test_cord, theta0)
        @test size(result) == (1, 5)
        @test all(isfinite, result)
    end

    # Verify compatibility with the training strategy loss pattern:
    # θ -> mean(abs2, loss_function(train_set, θ))
    for f in symbolic_loss.datafree_pde_loss_functions
        loss_val = mean(abs2, f(test_cord, theta0))
        @test isfinite(loss_val)
        @test loss_val >= 0
    end
end

@testitem "Symbolic PINN parser Zygote gradient compatibility" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
        heat_sys, chain; n_interior = 3, n_bc = 3
    )

    theta0 = symbolic_loss.theta0

    # Verify that Zygote can compute the gradient of the full loss
    grad = Zygote.gradient(symbolic_loss.loss, theta0)

    @test grad !== nothing
    @test length(grad) == 1
    @test all(isfinite, grad[1])
end
