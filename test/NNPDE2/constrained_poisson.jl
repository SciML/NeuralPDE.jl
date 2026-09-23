include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

using ADTypes: AutoForwardDiff
using Ipopt, OptimizationIpopt

@testset "2D Poisson with exact boundary constraints" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    analytic(x, y) = sinpi(x) * sinpi(y) / (2pi^2)
    chain = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
    xs = 0:0.05:1
    U_exact = [analytic(xi, yi) for xi in xs, yi in xs]

    ipopt(prob) = solve(
        prob, IpoptOptimizer(); maxiters = 500, reltol = 1.0e-8, verbose = 0
    )

    penalty_prob = discretize(
        pdesys,
        PhysicsInformedNN(
            chain, GridTraining(0.2); rng = Xoshiro(1173), boundary_policy = :penalty
        );
        adtype = AutoForwardDiff()
    )
    penalty_sol = ipopt(penalty_prob)
    penalty_error = maximum(
        abs, [penalty_sol(xi, yi; dv = u(x, y)) - analytic(xi, yi) for xi in xs, yi in xs]
    )

    constrained_prob = discretize(
        pdesys,
        PhysicsInformedNN(
            chain, GridTraining(0.2); rng = Xoshiro(1173), boundary_policy = :constraints
        );
        adtype = AutoForwardDiff()
    )
    # Ipopt started from the default init converges to u ≡ 0 on most seeds: it
    # satisfies the homogeneous constraints exactly, so it is a degenerate
    # stationary point. Warm-starting from the penalty solution avoids it.
    constrained_sol = ipopt(remake(constrained_prob; u0 = penalty_sol.original_sol.u))
    U = [constrained_sol(xi, yi; dv = u(x, y)) for xi in xs, yi in xs]
    constrained_error = maximum(abs, U .- U_exact)

    @test constrained_sol.retcode == SciMLBase.ReturnCode.Success
    expected_constraints = sum(
        block.npoints for block in pinn_metadata(constrained_prob).blocks if block.kind == :bc
    )
    @test length(constrained_prob.lcons) == expected_constraints
    @test constrained_prob.lcons == zeros(expected_constraints)
    @test constrained_prob.ucons == zeros(expected_constraints)
    residual = similar(constrained_prob.lcons)
    constrained_prob.f.cons(residual, constrained_sol.original_sol.u, constrained_prob.p)
    @test maximum(abs, residual) < 1.0e-8

    @test penalty_sol.retcode in (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.MaxIters)
    @test penalty_prob.ucons === nothing
    @info "2D Poisson interior error" constrained_error penalty_error
    @test maximum(abs, U) > 0.5 * maximum(abs, U_exact)
    @test penalty_error < 0.1 * maximum(abs, U_exact)
    @test constrained_error < 0.1 * maximum(abs, U_exact)
    @test constrained_error <= 2 * penalty_error
end
