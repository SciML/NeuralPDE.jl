include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

using ADTypes: AutoForwardDiff
using OptimizationIpopt

@testset "2D Poisson with exact boundary constraints" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    # Manufactured solution u* = x^2/2 + x y, so Δu* = 1. The nonhomogeneous
    # Dirichlet data makes u ≡ 0 infeasible: under homogeneous conditions Ipopt
    # can converge to the trivial zero network, which satisfies every boundary
    # constraint exactly while fitting nothing.
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 1.0
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.5 + y, u(x, 0) ~ 0.5x^2, u(x, 1) ~ 0.5x^2 + x]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    analytic(x, y) = 0.5x^2 + x * y
    chain = Chain(Dense(2, 6, tanh), Dense(6, 6, tanh), Dense(6, 1))
    xs = 0:0.05:1
    U_exact = [analytic(xi, yi) for xi in xs, yi in xs]

    # acceptable_iter = 0 disables Ipopt's "solved to acceptable level" exit.
    # Note that OptimizationIpopt also maps Feasible_Point_Found to Success, so
    # the residual assertion below checks the measured violation rather than
    # relying on the solver status alone.
    ipopt(prob) = solve(
        prob,
        IpoptOptimizer(; acceptable_iter = 0, constr_viol_tol = 1.0e-9,
            nlp_scaling_method = "none");
        maxiters = 3000, reltol = 1.0e-9, verbose = 0
    )

    # GridTraining(0.1) puts 44 boundary and 121 interior residuals on the
    # 67-parameter network, so the penalty objective cannot drive every
    # boundary residual to zero while the constrained formulation holds each
    # one at the constraint tolerance.
    penalty_prob = discretize(
        pdesys,
        PhysicsInformedNN(
            chain, GridTraining(0.1); rng = Xoshiro(1173), boundary_policy = :penalty
        );
        adtype = AutoForwardDiff()
    )
    penalty_sol = ipopt(penalty_prob)

    constrained_prob = discretize(
        pdesys,
        PhysicsInformedNN(
            chain, GridTraining(0.1); rng = Xoshiro(1173), boundary_policy = :constraints
        );
        adtype = AutoForwardDiff()
    )
    # A cold constrained solve here converges to a boundary-infeasible point
    # (it hits the iteration cap with residuals around 1e-2 on Julia 1.10), so
    # the solve is initialised from the penalty optimum. That point is
    # boundary-infeasible, so the constrained solve still does real work.
    constrained_sol = ipopt(remake(constrained_prob; u0 = penalty_sol.original_sol.u))

    bc_residual(θ) = begin
        r = zeros(length(constrained_prob.lcons))
        constrained_prob.f.cons(r, θ, constrained_prob.p)
        r
    end
    penalty_bc = maximum(abs, bc_residual(penalty_sol.original_sol.u))
    constrained_bc = maximum(abs, bc_residual(constrained_sol.original_sol.u))
    U = [constrained_sol(xi, yi; dv = u(x, y)) for xi in xs, yi in xs]
    constrained_error = maximum(abs, U .- U_exact)
    penalty_error = maximum(
        abs, [penalty_sol(xi, yi; dv = u(x, y)) - analytic(xi, yi) for xi in xs, yi in xs]
    )

    # The underparameterised penalty solve exhausts maxiters without reaching
    # strict optimality; its achieved boundary residual is the baseline here.
    @test penalty_sol.retcode ∈ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.MaxIters)
    @test constrained_sol.retcode == SciMLBase.ReturnCode.Success
    @test constrained_sol.stats.iterations > 0
    expected_constraints = sum(
        block.npoints for block in pinn_metadata(constrained_prob).blocks if block.kind == :bc
    )
    @test length(constrained_prob.lcons) == expected_constraints
    @test constrained_prob.lcons == zeros(expected_constraints)
    @test constrained_prob.ucons == zeros(expected_constraints)
    @test penalty_prob.ucons === nothing
    @test constrained_bc < 1.0e-7
    @test penalty_bc > 10 * constrained_bc
    @info "2D Poisson boundary feasibility and interior error" constrained_bc penalty_bc constrained_error penalty_error
    @test maximum(abs, U) > 0.5 * maximum(abs, U_exact)
    @test penalty_error < 0.1 * maximum(abs, U_exact)
    @test constrained_error < 0.1 * maximum(abs, U_exact)
    @test constrained_error < penalty_error
end
