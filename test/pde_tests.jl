@testsetup module NNPDE1TestSetup
using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers, QuasiMonteCarlo
import DomainSets: Interval

function callback(p, l)
    if p.iter == 1 || p.iter % 10 == 0
        println("Current loss is: $l after $(p.iter) iterations")
    end
    return false
end

strategies = [
    GridTraining(0.05),
    StochasticTraining(100),
    QuasiRandomTraining(100; sampling_alg = LatinHypercubeSample()),
]

export callback, strategies
end

@testitem "Heat Equation Basic Functionality" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux
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

    res_data = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; n_interior = 8, n_bc = 8)
    @test res_data.theta0 !== nothing
    @test length(res_data.pde_residuals) == 1
    @test length(res_data.bc_residuals) == 3
    @test length(res_data.datafree_pde_loss_functions) == 1
    @test length(res_data.datafree_bc_loss_functions) == 3

    l_init = res_data.loss(res_data.theta0)
    @test isfinite(l_init)
    @test l_init >= 0
end

@testitem "Coordinate Dynamism" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux
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
    sym_loss = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; n_interior = 4, n_bc = 4)
    theta0 = sym_loss.theta0
    f = sym_loss.datafree_pde_loss_functions[1]

    cord_a = rand(2, 8)
    cord_b = rand(2, 8)
    @test f(cord_a, theta0) != f(cord_b, theta0)

    disc_stoch = PhysicsInformedNN(chain, StochasticTraining(30))
    prob_stoch = discretize(heat_sys, disc_stoch)
    loss1 = prob_stoch.f(prob_stoch.u0, nothing)
    loss2 = prob_stoch.f(prob_stoch.u0, nothing)
    @test loss1 != loss2

    disc_grid = PhysicsInformedNN(chain, GridTraining(0.25))
    prob_grid = discretize(heat_sys, disc_grid)
    @test prob_grid.f(prob_grid.u0, nothing) == prob_grid.f(prob_grid.u0, nothing)
end

@testitem "Residual Correctness" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Symbolics
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))

    res_data = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain)
    pde_res = res_data.pde_residuals[1]

    @test pde_res isa Num
    raw_expr = Symbolics.unwrap(pde_res)
    @test raw_expr !== nothing
end

@testitem "Datafree Loss Function Format" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))
    res_data = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; n_interior = 4, n_bc = 4)

    cord = rand(2, 5)
    pde_loss_fn = res_data.datafree_pde_loss_functions[1]
    res_mat = pde_loss_fn(cord, res_data.theta0)

    @test res_mat isa Matrix
    @test size(res_mat) == (1, 5)
    @test all(isfinite, res_mat)
end

@testitem "Zygote Gradient Compatibility" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))
    res_data = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; n_interior = 4, n_bc = 4)

    loss_fn = res_data.loss
    θ0 = res_data.theta0

    grad = Zygote.gradient(loss_fn, θ0)
    @test grad !== nothing
    @test length(grad) == 1
    @test length(grad[1]) == length(θ0)
    @test all(isfinite, grad[1])
end

@testitem "Mixed Derivatives" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    using Test
    import DomainSets: Interval

    @parameters x y
    @variables u(..)
    Dxy = Differential(x) * Differential(y)

    eq = Dxy(u(x, y)) ~ 0.0
    bcs = [u(0.0, y) ~ y, u(x, 0.0) ~ x]
    domains = [x in Interval(0.0, 1.0), y in Interval(0.0, 1.0)]
    @named mixed_sys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))
    res_data = NeuralPDE.build_symbolic_pinn_loss(mixed_sys, chain; n_interior = 4, n_bc = 4)

    @test res_data.theta0 !== nothing
    l_init = res_data.loss(res_data.theta0)
    @test isfinite(l_init)
    @test l_init >= 0

    grad = Zygote.gradient(res_data.loss, res_data.theta0)
    @test grad !== nothing
    @test all(isfinite, grad[1])
end

@testitem "Single-Pass Prewalk Substitution" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Symbolics, SymbolicUtils
    using Test
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
    symbolic_loss = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; n_interior = 3, n_bc = 3)
    parsed = symbolic_loss.parsed
    spec   = symbolic_loss.neural_specs[1]

    @test all(r -> !NeuralPDE._contains_dv_call(r, parsed.dvs), symbolic_loss.pde_residuals)
    @test all(r -> !NeuralPDE._contains_dv_call(r, parsed.dvs), symbolic_loss.bc_residuals)

    function count_op_calls(expr, target_op)
        n = Ref(0)
        function walk(ex)
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
    @test count_op_calls(pde_res, spec.value) == 5

    for bc_res in symbolic_loss.bc_residuals
        @test count_op_calls(bc_res, spec.value) == 1
    end
end

@testitem "Multiple Dependent Variables" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t
    @variables u1(..) u2(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eqs = [Dt(u1(x, t)) ~ Dx(u2(x, t)), Dt(u2(x, t)) ~ Dx(u1(x, t))]
    bcs = [u1(0.0, t) ~ 0.0, u2(0.0, t) ~ 0.0, u1(x, 0.0) ~ sin(pi * x), u2(x, 0.0) ~ cos(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named coupled_sys = PDESystem(eqs, bcs, domains, [x, t], [u1(x, t), u2(x, t)])

    chains = [Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1)), Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))]
    res_data = NeuralPDE.build_symbolic_pinn_loss(coupled_sys, chains; n_interior = 4, n_bc = 4)
    @test length(res_data.pde_residuals) == 2
    @test length(res_data.bc_residuals) == 4

    l_init = res_data.loss(res_data.theta0)
    @test isfinite(l_init)

    grad = Zygote.gradient(res_data.loss, res_data.theta0)
    @test grad !== nothing
    @test all(isfinite, grad[1])
end

@testitem "Discretize Integration" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers
    using Test
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

    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    prob = discretize(heat_sys, discretization)
    @test prob isa Optimization.OptimizationProblem

    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)
    @test loss_val >= 0

    initial_loss = prob.f(prob.u0, nothing)
    sol = solve(prob, OptimizationOptimisers.Adam(0.02); maxiters = 500)
    final_loss = prob.f(sol.u, nothing)
    @test isfinite(final_loss)
    @test final_loss < initial_loss
end

@testitem "Training Strategies" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization
    using Test
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

    for strat in (GridTraining(0.25), StochasticTraining(50), QuasiRandomTraining(50))
        disc = PhysicsInformedNN(chain, strat)
        prob = discretize(heat_sys, disc)
        @test prob isa Optimization.OptimizationProblem
        @test isfinite(prob.f(prob.u0, nothing))
    end
end

@testitem "Discretize Zygote Gradient" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Zygote
    using Test
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
    discretization = PhysicsInformedNN(chain, GridTraining(0.25))
    prob = discretize(heat_sys, discretization)

    grad = Zygote.gradient(θ -> prob.f(θ, nothing), prob.u0)
    @test grad !== nothing
    @test all(isfinite, grad[1])
end

@testitem "Equation Parameter Support" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote
    using Test
    import DomainSets: Interval

    @parameters x t α
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ α * Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_param_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)], [α], initial_conditions = Dict([α => 1.0]))

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))

    disc_fixed = PhysicsInformedNN(chain, GridTraining(0.25))
    prob_fixed = discretize(heat_param_sys, disc_fixed)
    @test prob_fixed isa Optimization.OptimizationProblem
    @test isfinite(prob_fixed.f(prob_fixed.u0, nothing))

    disc_estim = PhysicsInformedNN(chain, GridTraining(0.25); param_estim = true)
    prob_estim = discretize(heat_param_sys, disc_estim)
    @test hasproperty(prob_estim.u0, :p)
    @test isfinite(prob_estim.f(prob_estim.u0, nothing))
end

@testitem "Finite Differences Evaluation" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote
    using Test
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
    discretization_sym = PhysicsInformedNN(chain, GridTraining(0.25))
    prob_sym = discretize(heat_sys, discretization_sym)

    @test prob_sym isa Optimization.OptimizationProblem
    @test isfinite(prob_sym.f(prob_sym.u0, nothing))
    grad_sym = Zygote.gradient(θ -> prob_sym.f(θ, nothing), prob_sym.u0)[1]
    @test grad_sym !== nothing
    @test all(isfinite, grad_sym)
end

@testitem "Training Strategies Gradients" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote, QuasiMonteCarlo, Integrals
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))

    for strat in (
        StochasticTraining(16),
        QuasiRandomTraining(16; sampling_alg = LatinHypercubeSample(), resampling = true),
        QuadratureTraining(; reltol = 1e-2, abstol = 1e-2, maxiters = 100),
    )
        discr = PhysicsInformedNN(chain, strat)
        prob = discretize(heat_sys, discr)
        @test isfinite(prob.f(prob.u0, nothing))
        grad = Zygote.gradient(θ -> prob.f(θ, nothing), prob.u0)
        @test grad !== nothing
        @test all(isfinite, grad[1])
    end
end

@testitem "Loss Expression Helper" tags = [:pinnparser] begin
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

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))
    exprs = symbolic_pinn_loss_expression(heat_sys, chain)

    @test length(exprs.pde) == 1
    @test length(exprs.bc) == 3
    @test isequal(exprs.ivs, [x, t])
    @test isequal(exprs.dvs, [u(x, t)])
end

@testitem "Loss Weighting" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, Zygote, Test
    import DomainSets: Interval

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq  = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named heat_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))

    sym_unweighted = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain)
    sym_weighted = NeuralPDE.build_symbolic_pinn_loss(heat_sys, chain; pde_loss_weights = 2.0, bc_loss_weights = 5.0)
    θ0 = sym_unweighted.theta0
    @test sym_weighted.pde_loss(θ0) ≈ 2.0 * sym_unweighted.pde_loss(θ0)
    @test sym_weighted.bc_loss(θ0) ≈ 5.0 * sym_unweighted.bc_loss(θ0)
end

@testitem "2D Poisson Equation Convergence" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Random, Test
    import DomainSets: Interval

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)
    bcs = [u(0.0, y) ~ 0.0, u(1.0, y) ~ 0.0, u(x, 0.0) ~ 0.0, u(x, 1.0) ~ 0.0]
    domains = [x in Interval(0.0, 1.0), y in Interval(0.0, 1.0)]
    @named poisson_sys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    chain = Lux.Chain(Lux.Dense(2, 12, tanh), Lux.Dense(12, 12, tanh), Lux.Dense(12, 1))

    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    prob = discretize(poisson_sys, discretization)
    @test prob isa Optimization.OptimizationProblem

    sol = solve(prob, OptimizationOptimisers.Adam(0.02); maxiters = 600)
    phi = discretization.phi

    xs = 0.1:0.2:0.9
    ys = 0.1:0.2:0.9
    analytic_sol(x, y) = sin(pi * x) * sin(pi * y) / (2 * pi^2)
    u_predict = [first(phi([x, y], sol.u)) for x in xs for y in ys]
    u_real = [analytic_sol(x, y) for x in xs for y in ys]

    @test u_predict ≈ u_real atol = 0.08
end

@testitem "1D Wave Equation Convergence" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Random, Test
    import DomainSets: Interval

    Random.seed!(100)

    @parameters x t
    @variables u(..)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    Dt = Differential(t)

    C = 1.0
    eq = Dtt(u(x, t)) ~ C^2 * Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x), Dt(u(x, 0.0)) ~ 0.0]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named wave_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 12, tanh), Lux.Dense(12, 12, tanh), Lux.Dense(12, 1))

    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    prob = discretize(wave_sys, discretization)

    initial_loss = prob.f(prob.u0, nothing)
    sol = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 1000)
    final_loss = prob.f(sol.u, nothing)

    @test final_loss < initial_loss
end

@testitem "1D Heterogeneous ODE Convergence" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Test
    import DomainSets: Interval

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    eq = Dθ(u(θ)) ~ θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) - u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))
    bcs = [u(0.0) ~ 1.0f0]
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    chain = Lux.Chain(Lux.Dense(1, 12, σ), Lux.Dense(12, 1))

    discretization = PhysicsInformedNN(chain, GridTraining(0.05))
    @named ode_sys = PDESystem(eq, bcs, domains, [θ], [u(θ)])

    prob = discretize(ode_sys, discretization)
    sol = solve(prob, OptimizationOptimisers.Adam(0.02); maxiters = 500)

    phi = discretization.phi
    analytic_ode(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = 0.1:0.1:0.9
    u_real = [analytic_ode(t) for t in ts]
    u_predict = [first(phi([t], sol.u)) for t in ts]

    @test u_predict ≈ u_real atol = 0.25
end

@testitem "3D Poisson Equation Convergence" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Random, Test
    import DomainSets: Interval

    Random.seed!(100)

    @parameters x y z
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2

    eq = Dxx(u(x, y, z)) + Dyy(u(x, y, z)) + Dzz(u(x, y, z)) ~ -3.0f0 * (pi^2) * sin(pi * x) * sin(pi * y) * sin(pi * z)
    bcs = [u(0.0, y, z) ~ 0.0, u(1.0, y, z) ~ 0.0, u(x, 0.0, z) ~ 0.0, u(x, 1.0, z) ~ 0.0, u(x, y, 0.0) ~ 0.0, u(x, y, 1.0) ~ 0.0]
    domains = [x in Interval(0.0, 1.0), y in Interval(0.0, 1.0), z in Interval(0.0, 1.0)]
    @named poisson3d_sys = PDESystem(eq, bcs, domains, [x, y, z], [u(x, y, z)])

    chain = Lux.Chain(Lux.Dense(3, 16, tanh), Lux.Dense(16, 16, tanh), Lux.Dense(16, 1))

    discretization = PhysicsInformedNN(chain, GridTraining(0.2))
    prob = discretize(poisson3d_sys, discretization)
    @test prob isa Optimization.OptimizationProblem

    initial_loss = prob.f(prob.u0, nothing)
    sol = solve(prob, OptimizationOptimisers.Adam(0.02); maxiters = 600)
    final_loss = prob.f(sol.u, nothing)
    @test final_loss < initial_loss
end

@testitem "1D Viscous Burgers Equation" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Random, Test
    import DomainSets: Interval

    Random.seed!(100)

    @parameters x t ν
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) ~ ν * Dxx(u(x, t))
    bcs = [u(-1.0, t) ~ 0.0, u(1.0, t) ~ 0.0, u(x, 0.0) ~ -sin(pi * x)]
    domains = [x in Interval(-1.0, 1.0), t in Interval(0.0, 1.0)]
    @named burger_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)], [ν], initial_conditions = Dict([ν => 0.01 / pi]))

    chain = Lux.Chain(Lux.Dense(2, 12, tanh), Lux.Dense(12, 12, tanh), Lux.Dense(12, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    prob = discretize(burger_sys, discretization)

    @test prob isa Optimization.OptimizationProblem
    initial_loss = prob.f(prob.u0, nothing)
    @test isfinite(initial_loss)

    sol = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 500)
    final_loss = prob.f(sol.u, nothing)
    @test isfinite(final_loss)
    @test final_loss < initial_loss
end

@testitem "Robin / Neumann Boundary Conditions" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Random, Test
    import DomainSets: Interval

    Random.seed!(100)

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(0.0, t) ~ 0.0, Dx(u(1.0, t)) + u(1.0, t) ~ 0.0, u(x, 0.0) ~ sin(pi * x)]
    domains = [x in Interval(0.0, 1.0), t in Interval(0.0, 1.0)]
    @named robin_sys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    chain = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    prob = discretize(robin_sys, discretization)

    @test prob isa Optimization.OptimizationProblem
    initial_loss = prob.f(prob.u0, nothing)
    @test isfinite(initial_loss)

    sol = solve(prob, OptimizationOptimisers.Adam(0.02); maxiters = 300)
    final_loss = prob.f(sol.u, nothing)
    @test isfinite(final_loss)
    @test final_loss < initial_loss
end

@testitem "3rd-Order Differential Operator" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Zygote, Test
    import DomainSets: Interval

    @parameters x
    @variables u(..)
    Dxxx = Differential(x)^3

    eq = Dxxx(u(x)) ~ cospi(x)
    bcs = [u(0.0) ~ 0.0, u(1.0) ~ 0.0, Differential(x)(u(0.0)) ~ 0.0]
    domains = [x in Interval(0.0, 1.0)]
    @named ode3_sys = PDESystem(eq, bcs, domains, [x], [u(x)])

    chain = Lux.Chain(Lux.Dense(1, 12, tanh), Lux.Dense(12, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.05))
    prob = discretize(ode3_sys, discretization)

    @test prob isa Optimization.OptimizationProblem
    loss_val = prob.f(prob.u0, nothing)
    @test isfinite(loss_val)

    grad = Zygote.gradient(θ -> prob.f(θ, nothing), prob.u0)[1]
    @test grad !== nothing
    @test all(isfinite, grad)
end

@testitem "Coupled First-Order PDE System" tags = [:pinnparser] begin
    using NeuralPDE, ModelingToolkit, DomainSets, Lux, Optimization, OptimizationOptimisers, Random, Test
    import DomainSets: Interval

    Random.seed!(100)

    @parameters x y
    @variables u1(..) u2(..)
    Dx = Differential(x)
    Dy = Differential(y)

    eqs = [Dx(u1(x, y)) + 4 * Dy(u2(x, y)) ~ 0, Dx(u2(x, y)) + 9 * Dy(u1(x, y)) ~ 0]
    bcs = [u1(x, 0.0) ~ 2 * x, u2(x, 0.0) ~ 3 * x]
    domains = [x in Interval(0.0, 1.0), y in Interval(0.0, 1.0)]
    @named coupled_sys = PDESystem(eqs, bcs, domains, [x, y], [u1(x, y), u2(x, y)])

    chains = [Lux.Chain(Lux.Dense(2, 12, tanh), Lux.Dense(12, 1)), Lux.Chain(Lux.Dense(2, 12, tanh), Lux.Dense(12, 1))]
    discretization = PhysicsInformedNN(chains, GridTraining(0.1))
    prob = discretize(coupled_sys, discretization)

    @test prob isa Optimization.OptimizationProblem
    initial_loss = prob.f(prob.u0, nothing)
    @test isfinite(initial_loss)

    sol = solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 400)
    final_loss = prob.f(sol.u, nothing)
    @test isfinite(final_loss)
    @test final_loss < initial_loss
end

@testitem "Test Heterogeneous ODE" tags = [:pinnparser] setup = [NNPDE1TestSetup] begin
    using Cubature, Integrals, QuasiMonteCarlo, DomainSets, Lux, Random, Optimisers
    import DomainSets: Interval, infimum, supremum

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
        @named pde_system = PDESystem(eq, bcs, domains, [θ], [u(θ)])
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



@testitem "PDE II - 2D Poisson" tags = [:pinnparser] setup = [NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS
    using LineSearches: BackTracking

    function test_2d_poisson_equation(chain, strategy)
        @parameters x y
        @variables u(..)
        Dxx = Differential(x)^2
        Dyy = Differential(y)^2

        # 2D PDE
        eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

        # Boundary conditions
        bcs = [
            u(0, y) ~ 0.0,
            u(1, y) ~ 0.0,
            u(x, 0) ~ 0.0,
            u(x, 1) ~ 0.0,
        ]

        # Space and time domains
        domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

        ps = Lux.initialparameters(Random.default_rng(), chain)

        discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
        @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
        prob = discretize(pde_system, discretization)
        res = solve(prob, Adam(0.01); maxiters = 1000, callback)
        prob = remake(prob, u0 = res.u)
        res = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 1000)
        phi = discretization.phi

        xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
        analytic = (x, y) -> (sinpi(x) * sinpi(y)) / (2pi^2)

        u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
        u_real = [analytic(x, y) for x in xs for y in ys]

        @test u_predict ≈ u_real atol = 2.0
    end

    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))

    @testset "$(nameof(typeof(strategy)))" for strategy in strategies
        test_2d_poisson_equation(chain, strategy)
    end

    algs = [CubatureJLp()]
    @testset "$(nameof(typeof(alg)))" for alg in algs
        strategy = QuadratureTraining(
            quadrature_alg = alg, reltol = 1.0e-4,
            abstol = 1.0e-3, maxiters = 30, batch = 10
        )
        test_2d_poisson_equation(chain, strategy)
    end
end

@testitem "PDE III - 3rd-order ODE" tags = [:pinnparser] setup = [NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
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
    quasirandom_strategy = QuasiRandomTraining(100; sampling_alg = LatinHypercubeSample())

    discretization = PhysicsInformedNN(chain, quasirandom_strategy)

    @named pde_system = PDESystem(
        eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)]
    )

    prob = discretize(pde_system, discretization)

    res = solve(prob, BFGS(); maxiters = 1000, callback)
    phi = discretization.phi[1]

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u.depvar.u)) for x in xs]

    @test u_predict ≈ u_real atol = 10^-4
end

@testitem "PDE IV - System of PDEs" tags = [:pinnparser] setup = [NNPDE1TestSetup] begin
    using Lux, Random, Optimisers, DomainSets, Cubature, QuasiMonteCarlo, Integrals
    import DomainSets: Interval, infimum, supremum

    @parameters x, y
    @variables u1(..), u2(..)
    Dx = Differential(x)
    Dy = Differential(y)

    # System of pde
    eqs = [
        Dx(u1(x, y)) + 4 * Dy(u2(x, y)) ~ 0,
        Dx(u2(x, y)) + 9 * Dy(u1(x, y)) ~ 0,
    ]

    # Initial and boundary conditions
    bcs = [
        u1(x, 0) ~ 2 * x,
        u2(x, 0) ~ 3 * x,
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # Neural network
    chain1 = Chain(Dense(2, 15, tanh), Dense(15, 1))
    chain2 = Chain(Dense(2, 15, tanh), Dense(15, 1))

    quadrature_strategy = QuadratureTraining(
        quadrature_alg = CubatureJLh(),
        reltol = 1.0e-3, abstol = 1.0e-3, maxiters = 50, batch = 100
    )
    chain = [chain1, chain2]

    discretization = PhysicsInformedNN(chain, quadrature_strategy)

    @named pde_system = PDESystem(eqs, bcs, domains, [x, y], [u1(x, y), u2(x, y)])

    prob = discretize(pde_system, discretization)

    res = solve(prob, Adam(0.01); maxiters = 2000, callback)
    phi = discretization.phi

    analytic_sol_func(x, y) = [1 / 3 * (6x - y), 1 / 2 * (6x - y)]
    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    u_real = [[analytic_sol_func(x, y)[i] for x in xs for y in ys] for i in 1:2]
    depvars = [:u1, :u2]

    u_predict = [
        [phi[i]([x, y], res.u.depvar[depvars[i]])[1] for x in xs for y in ys]
            for i in 1:2
    ]

    @test u_predict[1] ≈ u_real[1] atol = 0.3 norm = Base.Fix1(maximum, abs)
    @test u_predict[2] ≈ u_real[2] atol = 0.3 norm = Base.Fix1(maximum, abs)
end

@testitem "Fokker-Planck Equation" tags = [:pinnparser] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, ComponentArrays, Integrals, Cubature
    import DomainSets: Interval, infimum, supremum
    using OptimizationOptimJL: BFGS, LBFGS

    @parameters x
    @variables p(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    α, β, _σ = 0.3, 0.5, 0.5
    dx = 0.01

    eq = [Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))]
    x_0, x_end = -2.2, 2.2

    bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]
    domains = [x ∈ Interval(-2.2, 2.2)]

    inn = 18
    chain = Chain(Dense(1, inn, σ), Dense(inn, inn, σ), Dense(inn, inn, σ), Dense(inn, 1))

    init_params = ComponentArray{Float64}(
        Lux.initialparameters(Random.default_rng(), chain)
    )

    lb, ub = [x_0], [x_end]

    function norm_loss_function(phi, θ, p)
        inner_f(x, θ) = dx * phi(x, θ) .- 1
        prob1 = IntegralProblem(inner_f, (lb, ub), θ)
        norm2 = solve(prob1, HCubatureJL(), reltol = 1.0e-8, abstol = 1.0e-8, maxiters = 10)
        return abs(norm2[1])
    end

    discretization = PhysicsInformedNN(chain, GridTraining(dx); init_params, additional_loss = norm_loss_function)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
    prob = discretize(pde_system, discretization)
    phi = discretization.phi

    res = solve(prob, LBFGS(); maxiters = 400)
    prob = remake(prob; u0 = res.u)
    res = solve(prob, BFGS(); maxiters = 2000)

    C = 142.88418699042
    analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
    xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]

    u_predict = [first(phi(x, res.u)) for x in xs]
    @test u_predict ≈ u_real rtol = 1.0e-3
end

@testitem "Direct Function Approximation 1D" tags = [:pinnparser] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x
    @variables u(..)

    func(x) = @. 2 + abs(x - 0.5)

    eq = [u(x) ~ func(x)]
    bc = [u(0) ~ u(0)]

    x0 = 0
    x_end = 2
    dx = 0.001
    domain = [x ∈ Interval(x0, x_end)]

    xs = collect(x0:dx:x_end)

    hidden = 10
    chain = Chain(Dense(1, hidden, tanh), Dense(hidden, hidden, tanh), Dense(hidden, 1))

    strategy = GridTraining(0.01)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(0.05), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(initial_stepnorm = 0.01), maxiters = 500)

    @test discretization.phi(xs', res.u) ≈ func(xs') rtol = 0.02
end

@testitem "Direct Function Approximation 2D" tags = [:pinnparser] begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x, y
    @variables u(..)
    func(x, y) = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))

    eq = [u(x, y) ~ func(x, y)]
    bc = [u(0, 0) ~ u(0, 0)]

    x0 = -10
    x_end = 10
    y0 = -10
    y_end = 10
    d = 0.4
    domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]
    hidden = 25
    chain = Chain(
        Dense(2, hidden, tanh), Dense(hidden, hidden, tanh),
        Dense(hidden, hidden, tanh), Dense(hidden, 1)
    )
    strategy = GridTraining(d)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 500)
    phi = discretization.phi
    xs = collect(x0:0.1:x_end)
    ys = collect(y0:0.1:y_end)
    u_predict = reshape(
        [first(phi([x, y], res.u)) for x in xs for y in ys],
        (length(xs), length(ys))
    )
    u_real = reshape([func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
    @test u_predict ≈ u_real rtol = 0.05
end

@testitem "Trivial BC [0 ~ 0] fails for some training strategies" tags = [:pinnparser] begin
    using NeuralPDE, Lux, ModelingToolkit, Symbolics, Optimization, Test
    import DomainSets: Interval

    @parameters x
    @variables u(..)
    Dx = Differential(x)

    eq = Dx(u(x)) ~ 1.0
    bcs = Symbolics.Equation[0.0 ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(1, 8, σ), Dense(8, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))

    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    @test prob isa Optimization.OptimizationProblem
end

@testitem "Empty boundary condition [] fails in solve phase" tags = [:pinnparser] begin
    using NeuralPDE, Lux, ModelingToolkit, Symbolics, Optimization, Test
    import DomainSets: Interval

    @parameters x
    @variables u(..)
    Dx = Differential(x)

    eq = Dx(u(x)) ~ 1.0
    bcs = Symbolics.Equation[]
    domains = [x ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(1, 8, σ), Dense(8, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))

    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    @test prob isa Optimization.OptimizationProblem
    @test isfinite(prob.f(prob.u0, nothing))
end
