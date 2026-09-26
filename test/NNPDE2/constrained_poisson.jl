include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

using ADTypes: AutoForwardDiff
using OptimizationIpopt
import Ipopt

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

    # Fixed biquadratic features make both optimization problems convex quadratics;
    # the sinusoidal solution lies outside this trial space.
    legendre(z) = (one.(z), 2z .- 1, 6z .^ 2 .- 6z .+ 1)
    features(X) = vcat(
        [
            a .* b for a in legendre(X[1:1, :]) for b in legendre(X[2:2, :])
        ]...
    )
    chain = Chain(WrappedFunction(features), Dense(9, 1; use_bias = false))
    problem(policy) = discretize(
        pdesys,
        PhysicsInformedNN(
            chain, StochasticTraining(100; bcs_points = 2);
            rng = Xoshiro(1173), boundary_policy = policy
        );
        adtype = AutoForwardDiff()
    )
    penalty_prob = problem(:penalty)
    constrained_prob = problem(:constraints)
    constr_viol_tol = 1.0e-9
    ipopt(prob) = solve(
        prob,
        IpoptOptimizer(;
            acceptable_iter = 0, constr_viol_tol, nlp_scaling_method = "none"
        );
        maxiters = 100, reltol = 1.0e-8, verbose = 0
    )
    penalty_sol = ipopt(penalty_prob)
    constrained_sol = ipopt(constrained_prob)
    for sol in (penalty_sol, constrained_sol)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test Ipopt.ApplicationReturnStatus(sol.original_sol.original.status) == Ipopt.Solve_Succeeded
        @test sol.stats.iterations > 0
    end

    expected_constraints = sum(
        block.npoints for block in pinn_metadata(constrained_prob).blocks if block.kind == :bc
    )
    @test length(constrained_prob.lcons) == expected_constraints == 8
    @test constrained_prob.lcons == zeros(expected_constraints)
    @test constrained_prob.ucons == zeros(expected_constraints)
    @test penalty_prob.ucons === nothing
    bc_residual(θ) = begin
        r = zeros(length(constrained_prob.lcons))
        constrained_prob.f.cons(r, θ, constrained_prob.p)
        r
    end
    @test maximum(abs, bc_residual(constrained_prob.u0)) > constr_viol_tol
    penalty_bc = maximum(abs, bc_residual(penalty_sol.original_sol.u))
    constrained_bc = maximum(abs, bc_residual(constrained_sol.original_sol.u))
    @test constrained_bc <= constr_viol_tol
    @test penalty_bc > constr_viol_tol

    xs = 0.05:0.05:0.95
    U_exact = [analytic(xi, yi) for xi in xs, yi in xs]
    error(sol) = maximum(
        abs, [sol(xi, yi; dv = u(x, y)) for xi in xs, yi in xs] .- U_exact
    )
    penalty_error = error(penalty_sol)
    constrained_error = error(constrained_sol)
    @test constrained_error < maximum(abs, U_exact)
    @test constrained_error < penalty_error
    @info "2D Poisson boundary feasibility and interior error" constrained_bc penalty_bc constrained_error penalty_error
end
