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

    # Verify collocation points are stored as matrices
    @test symbolic_loss.points.pde isa Matrix{Float64}
    @test symbolic_loss.points.bc isa AbstractVector{<:Matrix{Float64}}
    @test size(symbolic_loss.points.pde, 1) == 2  # D = 2 (x, t)
    @test all(m -> size(m, 1) == 2, symbolic_loss.points.bc)

    # Verify batched loss matches manual point-by-point evaluation
    pde_fn = symbolic_loss.datafree_pde_loss_functions[1]
    pde_pts = symbolic_loss.points.pde
    manual_loss = sum(abs2, pde_fn(pde_pts, theta0)) / size(pde_pts, 2)
    @test pde_loss ≈ manual_loss
end

@testitem "Symbolic PINN parser coordinate dynamism" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])
    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    sym_loss = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; n_interior = 4, n_bc = 4)
    theta0 = sym_loss.theta0
    f = sym_loss.datafree_pde_loss_functions[1]

    # -----------------------------------------------------------------------
    # Test 1: same compiled fn, different coordinate matrices → different residuals.
    # Proves coordinates are NOT baked into the compiled expression.
    # -----------------------------------------------------------------------
    cord_a = rand(2, 8)
    cord_b = rand(2, 8)
    @test f(cord_a, theta0) != f(cord_b, theta0)

    # -----------------------------------------------------------------------
    # Test 2: StochasticTraining resamples on each call →
    # consecutive evaluations at the SAME theta give DIFFERENT loss values.
    # -----------------------------------------------------------------------
    disc_stoch = PhysicsInformedNN(
        chain, StochasticTraining(30); symbolic_parser = true
    )
    prob_stoch = discretize(heat_sys, disc_stoch)
    loss1 = prob_stoch.f(prob_stoch.u0, nothing)
    loss2 = prob_stoch.f(prob_stoch.u0, nothing)
    # Probability of exact equality for random Float64 coords is essentially zero
    @test loss1 != loss2

    # -----------------------------------------------------------------------
    # Test 3: GridTraining uses fixed points → identical loss on every call.
    # -----------------------------------------------------------------------
    disc_grid = PhysicsInformedNN(
        chain, GridTraining(0.25); symbolic_parser = true
    )
    prob_grid = discretize(heat_sys, disc_grid)
    @test prob_grid.f(prob_grid.u0, nothing) == prob_grid.f(prob_grid.u0, nothing)
end

@testitem "Symbolic PINN parser residual correctness" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Symbolics
    using Test
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

    parsed = symbolic_loss.parsed
    spec = symbolic_loss.neural_specs[1]

    # Helper function to count calls in a Symbolic expression
    function count_calls(expr, op)
        count = Ref(0)
        function walk(ex)
            SymbolicUtils.iscall(ex) || return
            if isequal(SymbolicUtils.operation(ex), op)
                count[] += 1
            end
            for arg in SymbolicUtils.arguments(ex)
                walk(arg)
            end
        end
        walk(Symbolics.unwrap(expr))
        return count[]
    end

    # 1. Verify symbolic structure (NN call counts)
    pde_res = symbolic_loss.pde_residuals[1]
    bc_res1 = symbolic_loss.bc_residuals[1]
    bc_res2 = symbolic_loss.bc_residuals[2]
    bc_res3 = symbolic_loss.bc_residuals[3]

    @test count_calls(pde_res, spec.value) == 5

    @test count_calls(bc_res1, spec.value) == 1
    @test count_calls(bc_res2, spec.value) == 1
    @test count_calls(bc_res3, spec.value) == 1

    # 2. Evaluate residual at analytical solution
    analytical_nn(input, p) = [exp(-pi^2 * input[2]) * sin(pi * input[1])]

    iv_args = Symbolics.unwrap.(parsed.ivs)
    nn_args = [spec.value]
    p_args = [spec.parameters]

    compiled_pde = Symbolics.build_function(
        pde_res, iv_args..., nn_args..., p_args...;
        expression = Val(false)
    )

    compiled_bcs = [
        Symbolics.build_function(
            res, iv_args..., nn_args..., p_args...;
            expression = Val(false)
        ) for res in symbolic_loss.bc_residuals
    ]

    theta0 = symbolic_loss.theta0
    test_points = [[0.25, 0.5], [0.5, 0.25], [0.75, 0.1]]

    for pt in test_points
        x_val, t_val = pt[1], pt[2]
        pde_val = compiled_pde(x_val, t_val, analytical_nn, theta0)[1]
        bc1_val = compiled_bcs[1](0.0, t_val, analytical_nn, theta0)[1]
        bc2_val = compiled_bcs[2](1.0, t_val, analytical_nn, theta0)[1]
        bc3_val = compiled_bcs[3](x_val, 0.0, analytical_nn, theta0)[1]

        @test pde_val ≈ 0.0 atol = 1e-6
        @test bc1_val ≈ 0.0 atol = 1e-6
        @test bc2_val ≈ 0.0 atol = 1e-6
        @test bc3_val ≈ 0.0 atol = 1e-6
    end
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

@testitem "Symbolic PINN parser mixed derivatives" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # PDE with a mixed derivative: ∂²u/∂x∂t + ∂²u/∂x² ~ 0
    eq = Dx(Dt(u(x, t))) + Dxx(u(x, t)) ~ 0
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named mixed_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
        mixed_sys, chain; n_interior = 3, n_bc = 3
    )

    theta0 = symbolic_loss.theta0
    @test !isempty(theta0)

    # Verify no DV calls remain in the residuals
    parsed = NeuralPDE.parse_pde_system(mixed_sys)
    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.pde_residuals
    )

    # Verify loss is finite
    pde_loss = symbolic_loss.pde_loss(theta0)
    bc_loss = symbolic_loss.bc_loss(theta0)
    full_loss = symbolic_loss.loss(theta0)
    @test isfinite(pde_loss)
    @test isfinite(bc_loss)
    @test isfinite(full_loss)
    @test pde_loss >= 0
    @test bc_loss >= 0
    @test full_loss == pde_loss + bc_loss

    # Verify Zygote gradient works through mixed derivatives
    grad = Zygote.gradient(symbolic_loss.loss, theta0)
    @test grad !== nothing
    @test length(grad) == 1
    @test all(isfinite, grad[1])
end

@testitem "Symbolic PINN parser single-pass prewalk substitution" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Symbolics, SymbolicUtils
    using Test
    import DomainSets: Interval

    # -----------------------------------------------------------------------
    # Setup: heat equation system
    # -----------------------------------------------------------------------
    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
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
    parsed = symbolic_loss.parsed
    spec   = symbolic_loss.neural_specs[1]

    # Helper: count nodes visited during a tree walk (to verify single-pass)
    function count_nodes(expr)
        n = Ref(0)
        function walk(ex)
            n[] += 1
            SymbolicUtils.iscall(ex) || return
            for arg in SymbolicUtils.arguments(ex)
                walk(arg)
            end
        end
        walk(Symbolics.unwrap(expr))
        return n[]
    end

    # -----------------------------------------------------------------------
    # Test 1: No raw DV calls survive in any residual (correctness)
    # -----------------------------------------------------------------------
    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.pde_residuals
    )
    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.bc_residuals
    )

    # -----------------------------------------------------------------------
    # Test 2: PDE residual has derivative (DNN) ops, not value (NN) ops
    # -----------------------------------------------------------------------
    function count_op_calls(expr, target_op)
        n = Ref(0)
        function walk(ex)
            # Skip non-symbolic types (e.g. concrete Int/Vector args inside DNN calls)
            ex isa SymbolicUtils.BasicSymbolic || return
            SymbolicUtils.iscall(ex) || return
            if isequal(SymbolicUtils.operation(ex), target_op)
                n[] += 1
            end
            for arg in SymbolicUtils.arguments(ex)
                walk(arg)
            end
        end
        walk(Symbolics.unwrap(expr))
        return n[]
    end

    pde_res = symbolic_loss.pde_residuals[1]
    # heat equation residual: ∂ₜu - ∂ₓₓu → 5 NN calls
    @test count_op_calls(pde_res, spec.value) == 5

    # -----------------------------------------------------------------------
    # Test 3: Each BC residual has exactly one NN call
    # -----------------------------------------------------------------------
    for bc_res in symbolic_loss.bc_residuals
        @test count_op_calls(bc_res, spec.value) == 1
    end

    # -----------------------------------------------------------------------
    # Test 4: Single-pass property – prewalk does NOT descend into a subtree
    # it has already substituted. We verify this by checking that the output
    # residuals contain no nested/redundant NN-inside-NN structure.
    # Concretely: every NN call argument must NOT contain another NN call.
    # -----------------------------------------------------------------------
    function nn_args_clean(expr, nn_op)
        expr isa SymbolicUtils.BasicSymbolic || return true
        SymbolicUtils.iscall(expr) || return true
        op = SymbolicUtils.operation(expr)
        if isequal(op, nn_op)
            # None of the arguments of this NN call should themselves
            # contain an NN call (that would indicate nested substitution).
            for arg in SymbolicUtils.arguments(expr)
                arg isa SymbolicUtils.BasicSymbolic || continue
                count_op_calls(arg, nn_op) == 0 || return false
            end
            return true  # stop here; don't recurse into already-substituted subtree
        end
        return all(arg -> nn_args_clean(arg, nn_op), SymbolicUtils.arguments(expr))
    end

    for res in symbolic_loss.pde_residuals
        @test nn_args_clean(Symbolics.unwrap(res), spec.value)
    end

    # -----------------------------------------------------------------------
    # Test 5: Non-DV subexpressions are left structurally unchanged.
    # The sin(pi*x) term in the IC residual should still be present.
    # -----------------------------------------------------------------------
    ic_res = symbolic_loss.bc_residuals[3]  # u(x, 0) ~ sin(pi*x)
    ic_expr = Symbolics.unwrap(ic_res)

    function contains_sin(ex)
        SymbolicUtils.iscall(ex) || return false
        isequal(SymbolicUtils.operation(ex), sin) && return true
        return any(contains_sin, SymbolicUtils.arguments(ex))
    end

    @test contains_sin(ic_expr)

    # -----------------------------------------------------------------------
    # Test 6: All residuals produce finite values after lowering (smoke test)
    # -----------------------------------------------------------------------
    theta0 = symbolic_loss.theta0
    for f in symbolic_loss.residual_functions.pde
        val = f([0.3, 0.4], theta0)
        @test isfinite(first(val))
    end
    for f in symbolic_loss.residual_functions.bc
        val = f([0.3, 0.4], theta0)
        @test isfinite(first(val))
    end

    # -----------------------------------------------------------------------
    # Test 7: Mixed-derivative PDE residual (∂ₓ∂ₜu) is handled in one pass
    # -----------------------------------------------------------------------
    @parameters x2 t2
    @variables v(..)
    Dt2  = Differential(t2)
    Dx2  = Differential(x2)
    Dxx2 = Differential(x2)^2

    eq_mixed = Dx2(Dt2(v(x2, t2))) + Dxx2(v(x2, t2)) ~ 0
    bcs_mixed = [
        v(0.0, t2) ~ 0.0,
        v(1.0, t2) ~ 0.0,
        v(x2, 0.0) ~ sin(pi * x2),
    ]
    domains_mixed = [x2 in Interval(0.0, 1.0), t2 in Interval(0.0, 1.0)]
    @named mixed_sys = PDESystem(eq_mixed, bcs_mixed, domains_mixed, [x2, t2], [v(x2, t2)])

    chain_m = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
    sym_loss_m = NeuralPDE.build_symbolic_pinn_loss(
        mixed_sys, chain_m; n_interior = 3, n_bc = 3
    )
    parsed_m = NeuralPDE.parse_pde_system(mixed_sys)

    @test all(
        r -> !NeuralPDE._contains_dv_call(r, parsed_m.dvs),
        sym_loss_m.pde_residuals
    )
    @test all(
        r -> !NeuralPDE._contains_dv_call(r, parsed_m.dvs),
        sym_loss_m.bc_residuals
    )

    theta0_m = sym_loss_m.theta0
    @test isfinite(sym_loss_m.pde_loss(theta0_m))
    @test isfinite(sym_loss_m.bc_loss(theta0_m))
end

@testitem "Symbolic PINN parser multiple dependent variables" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # Coupled system: two dependent variables
    eq1 = Dt(u(x, t)) ~ Dxx(u(x, t)) + v(x, t)
    eq2 = Dt(v(x, t)) ~ Dxx(v(x, t)) - u(x, t)
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        v(0.0, t) ~ 0.0,
        v(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
        v(x, 0.0) ~ 0.0,
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named coupled_sys = PDESystem(
        [eq1, eq2], bcs, domains, [x, t], [u(x, t), v(x, t)]
    )

    chain1 = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
    chain2 = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
        coupled_sys, [chain1, chain2]; n_interior = 3, n_bc = 3
    )

    theta0 = symbolic_loss.theta0

    # Verify theta0 contains params from both networks
    spec1_len = length(NeuralPDE._theta0(symbolic_loss.neural_specs[1]))
    spec2_len = length(NeuralPDE._theta0(symbolic_loss.neural_specs[2]))
    @test length(theta0) == spec1_len + spec2_len

    # Verify no DV calls remain in any residuals
    parsed = NeuralPDE.parse_pde_system(coupled_sys)
    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.pde_residuals
    )
    @test all(
        residual -> !NeuralPDE._contains_dv_call(residual, parsed.dvs),
        symbolic_loss.bc_residuals
    )

    # Verify loss is finite
    pde_loss = symbolic_loss.pde_loss(theta0)
    bc_loss = symbolic_loss.bc_loss(theta0)
    full_loss = symbolic_loss.loss(theta0)
    @test isfinite(pde_loss)
    @test isfinite(bc_loss)
    @test isfinite(full_loss)
    @test pde_loss >= 0
    @test bc_loss >= 0
    @test full_loss == pde_loss + bc_loss

    # Verify Zygote gradient works through multi-DV system
    grad = Zygote.gradient(symbolic_loss.loss, theta0)
    @test grad !== nothing
    @test length(grad) == 1
    @test all(isfinite, grad[1])
end

# ===================================================================
# Tier 2 Integration Tests
# ===================================================================

@testitem "Symbolic PINN parser discretize integration" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    # --- Test 1: discretize returns a valid OptimizationProblem ---
    discretization = PhysicsInformedNN(
        chain, GridTraining(0.1); symbolic_parser = true
    )
    prob = discretize(heat_sys, discretization)
    @test prob isa Optimization.OptimizationProblem

    # --- Test 2: The loss function evaluates to a finite value ---
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= 0

    # --- Test 3: Short optimization loop converges (loss decreases) ---
    initial_loss = prob.f(prob.u0, nothing)
    sol = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 20)
    final_loss = prob.f(sol.u, nothing)
    @test isfinite(final_loss)
    @test final_loss <= initial_loss
end

@testitem "Symbolic PINN parser training strategies" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    # --- Test 1: GridTraining ---
    disc_grid = PhysicsInformedNN(
        chain, GridTraining(0.25); symbolic_parser = true
    )
    prob_grid = discretize(heat_sys, disc_grid)
    @test prob_grid isa Optimization.OptimizationProblem
    loss_grid = prob_grid.f(prob_grid.u0, nothing)
    @test isfinite(loss_grid)
    @test loss_grid >= 0

    # --- Test 2: StochasticTraining ---
    disc_stoch = PhysicsInformedNN(
        chain, StochasticTraining(50); symbolic_parser = true
    )
    prob_stoch = discretize(heat_sys, disc_stoch)
    @test prob_stoch isa Optimization.OptimizationProblem
    loss_stoch = prob_stoch.f(prob_stoch.u0, nothing)
    @test isfinite(loss_stoch)
    @test loss_stoch >= 0

    # --- Test 3: QuasiRandomTraining ---
    disc_quasi = PhysicsInformedNN(
        chain, QuasiRandomTraining(50); symbolic_parser = true
    )
    prob_quasi = discretize(heat_sys, disc_quasi)
    @test prob_quasi isa Optimization.OptimizationProblem
    loss_quasi = prob_quasi.f(prob_quasi.u0, nothing)
    @test isfinite(loss_quasi)
    @test loss_quasi >= 0
end

@testitem "Symbolic PINN parser adaptive loss" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    # --- Test 1: NonAdaptiveLoss with custom weights ---
    disc = PhysicsInformedNN(
        chain, GridTraining(0.25);
        symbolic_parser = true,
        adaptive_loss = NonAdaptiveLoss(; pde_loss_weights = 2.0, bc_loss_weights = 1.0)
    )
    prob = discretize(heat_sys, disc)
    @test prob isa Optimization.OptimizationProblem
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= 0
end

@testitem "Symbolic PINN parser discretize Zygote gradient" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    discretization = PhysicsInformedNN(
        chain, GridTraining(0.25); symbolic_parser = true
    )
    prob = discretize(heat_sys, discretization)

    # Verify Zygote can differentiate through the full discretize-generated loss
    grad = Zygote.gradient(θ -> prob.f(θ, nothing), prob.u0)
    @test grad !== nothing
    @test length(grad) == 1
    @test length(grad[1]) == length(prob.u0)
    @test all(isfinite, grad[1])
end

@testitem "Symbolic PINN parser MiniMaxAdaptiveLoss" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    # Use MiniMaxAdaptiveLoss — requires gradient evaluation to reweight
    disc = PhysicsInformedNN(
        chain, GridTraining(0.25);
        symbolic_parser = true,
        adaptive_loss = MiniMaxAdaptiveLoss(10; pde_max_optimiser = OptimizationOptimisers.Adam(0.01),
                                              bc_max_optimiser = OptimizationOptimisers.Adam(0.01))
    )
    prob = discretize(heat_sys, disc)
    @test prob isa Optimization.OptimizationProblem

    # Evaluate loss multiple times to trigger adaptive reweighting
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= 0

    # Run a few iterations to exercise the adaptive loss machinery
    sol = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 10)
    @test isfinite(prob.f(sol.u, nothing))
end

@testitem "Symbolic PINN parser equation parameter support" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t α
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ α * Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_param_sys = PDESystem(
        eq,
        bcs,
        domains,
        [x, t],
        [u(x, t)],
        [α],
        initial_conditions = Dict([α => 1.0])
    )

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    # Fixed-parameter mode: uses defaults from initial_conditions.
    disc_fixed = PhysicsInformedNN(chain, GridTraining(0.25); symbolic_parser = true)
    prob_fixed = discretize(heat_param_sys, disc_fixed)
    @test prob_fixed isa Optimization.OptimizationProblem
    @test isfinite(prob_fixed.f(prob_fixed.u0, nothing))

    grad_fixed = Zygote.gradient(θ -> prob_fixed.f(θ, nothing), prob_fixed.u0)
    @test grad_fixed !== nothing
    @test all(isfinite, grad_fixed[1])

    # Parameter-estimation mode: α is carried in theta.p and participates in AD.
    disc_estim = PhysicsInformedNN(
        chain,
        GridTraining(0.25);
        symbolic_parser = true,
        param_estim = true,
    )
    prob_estim = discretize(heat_param_sys, disc_estim)
    @test prob_estim isa Optimization.OptimizationProblem
    @test hasproperty(prob_estim.u0, :p)
    @test length(prob_estim.u0.p) == 1

    loss_a = prob_estim.f(prob_estim.u0, nothing)
    @test isfinite(loss_a)

    θ2 = deepcopy(prob_estim.u0)
    θ2.p .= θ2.p .* 2
    loss_b = prob_estim.f(θ2, nothing)
    @test isfinite(loss_b)

    grad_estim = Zygote.gradient(θ -> prob_estim.f(θ, nothing), prob_estim.u0)
    @test grad_estim !== nothing
    @test all(isfinite, grad_estim[1].depvar)
    @test all(isfinite, grad_estim[1].p)
end

@testitem "Symbolic PINN parser (Finite Differences)" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    # Test build_symbolic_pinn_loss
    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(
        heat_sys, chain; n_interior = 3, n_bc = 3
    )
    theta0 = symbolic_loss.theta0
    @test isfinite(symbolic_loss.loss(theta0))
    grad = Zygote.gradient(symbolic_loss.loss, theta0)
    @test grad !== nothing
    @test all(isfinite, grad[1])

    # Test PhysicsInformedNN discretization
    discretization = PhysicsInformedNN(
        chain, GridTraining(0.25); symbolic_parser = true
    )
    prob = discretize(heat_sys, discretization)
    @test prob isa Optimization.OptimizationProblem
    
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)

    # Differentiate through the loss function
    grad2 = Zygote.gradient(θ -> prob.f(θ, nothing), prob.u0)
    @test grad2 !== nothing
    @test all(isfinite, grad2[1])
end

@testitem "Symbolic PINN parser training strategies gradients" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote, QuasiMonteCarlo, Integrals
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))

    # 1. StochasticTraining
    discr_stoch = PhysicsInformedNN(
        chain, StochasticTraining(16); symbolic_parser = true
    )
    prob_stoch = discretize(heat_sys, discr_stoch)
    @test isfinite(prob_stoch.f(prob_stoch.u0, nothing))
    grad_stoch = Zygote.gradient(θ -> prob_stoch.f(θ, nothing), prob_stoch.u0)
    @test grad_stoch !== nothing
    @test all(isfinite, grad_stoch[1])

    # 2. QuasiRandomTraining
    discr_quasi = PhysicsInformedNN(
        chain, QuasiRandomTraining(16; sampling_alg = LatinHypercubeSample(), resampling = true);
        symbolic_parser = true
    )
    prob_quasi = discretize(heat_sys, discr_quasi)
    @test isfinite(prob_quasi.f(prob_quasi.u0, nothing))
    grad_quasi = Zygote.gradient(θ -> prob_quasi.f(θ, nothing), prob_quasi.u0)
    @test grad_quasi !== nothing
    @test all(isfinite, grad_quasi[1])

    # 3. QuadratureTraining
    discr_quad = PhysicsInformedNN(
        chain, QuadratureTraining(; reltol = 1e-2, abstol = 1e-2, maxiters = 100);
        symbolic_parser = true
    )
    prob_quad = discretize(heat_sys, discr_quad)
    @test isfinite(prob_quad.f(prob_quad.u0, nothing))
    grad_quad = Zygote.gradient(θ -> prob_quad.f(θ, nothing), prob_quad.u0)
    @test grad_quad !== nothing
    @test all(isfinite, grad_quad[1])
end


@testitem "Symbolic PINN parser loss expression helper" tags = [:symbolicpinn] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [
        u(0.0, t) ~ 0.0,
        u(1.0, t) ~ 0.0,
        u(x, 0.0) ~ sin(pi * x),
    ]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))

    # Test the exposed helper
    exprs = symbolic_pinn_loss_expression(heat_sys, chain)

    @test length(exprs.pde) == 1
    @test length(exprs.bc) == 3
    @test isequal(exprs.ivs, [x, t])
    @test isequal(exprs.dvs, [u(x, t)])
end

