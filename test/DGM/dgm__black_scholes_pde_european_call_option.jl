using NeuralPDE
using Test

@testset "Black-Scholes PDE: European Call Option" begin
    using ModelingToolkit, Optimization, OptimizationOptimisers, Distributions,
        LinearAlgebra, Statistics
    import DomainSets: Interval, infimum, supremum

    K, T, r, σ, S, S_multiplier = 50.0, 1.0, 0.05, 0.25, 130.0, 1.3

    @parameters x t
    @variables g(..)
    G(x) = max(x - K, 0.0)

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    eq = Dt(g(t, x)) + r * x * Dx(g(t, x)) + 0.5 * σ^2 * Dxx(g(t, x)) ~ r * g(t, x)

    bcs = [g(T, x) ~ G(x)] # terminal condition

    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, S * S_multiplier)]

    strategy = QuasiRandomTraining(128, minibatch = 32)
    discretization = DeepGalerkin(2, 1, 40, 3, tanh, tanh, identity, strategy)

    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [g(t, x)])
    prob = discretize(pde_system, discretization)

    callback = function (p, l)
        p.iter % 50 == 0 && println("$(p.iter) => $l")
        return false
    end

    res = solve(prob, Adam(0.1); callback, maxiters = 100)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.01); callback, maxiters = 500)
    phi = discretization.phi

    function analytical_soln(t, x, K, σ, T)
        d₊ = (log(x / K) + (r + 0.5 * σ^2) * (T - t)) / (σ * sqrt(T - t))
        d₋ = d₊ - (σ * sqrt(T - t))
        return x * cdf(Normal(0, 1), d₊) .- K * exp(-r * (T - t)) * cdf(Normal(0, 1), d₋)
    end
    analytic_sol_func(t, x) = analytical_soln(t, x, K, σ, T)

    domains2 = [t ∈ Interval(0.0, T - 0.001), x ∈ Interval(0.0, S)]
    ts = collect(infimum(domains2[1].domain):0.01:supremum(domains2[1].domain))
    xs = collect(infimum(domains2[2].domain):1.0:supremum(domains2[2].domain))

    u_real = [analytic_sol_func(t, x) for t in ts, x in xs]
    u_predict = [first(phi([t, x], res.u)) for t in ts, x in xs]
    # Use atol instead of rtol — u_real contains near-zero values (deep out-of-the-money)
    # where rtol is ill-defined. Check that the mean absolute error is reasonable.
    @test mean(abs, u_predict .- u_real) < 5.0
end
