@testitem "OU process" tags = [:nnsde2] begin
    using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Optimisers
    using OrdinaryDiffEq, Random, Distributions, Integrals, Cubature
    using DifferentialEquations, LineSearches
    using ModelingToolkit: infimum, supremum
    using OptimizationOptimJL: BFGS
    Random.seed!(100)

    α = -1
    β = 1
    u0 = 0.5
    t0 = 0.0
    f(u, p, t) = α * u
    g(u, p, t) = β
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u0, tspan)

    # Neural network
    inn = 20
    chain = Lux.Chain(Dense(2, inn, Lux.tanh),
        Dense(inn, inn, Lux.tanh),
        Dense(inn, 1, Lux.logcosh
        )) |> f64

    # problem setting
    dx = 0.01
    x_0 = -4.0
    x_end = 4.0
    σ_var_bc = 0.05

    alg = SDEPINN(
        chain=chain,
        optimalg=BFGS(),
        norm_loss_alg=HCubatureJL(),
        x_0=x_0,
        x_end=x_end,
        distrib=Normal(u0, σ_var_bc)
    )

    sol_OU, phi = solve(
        prob,
        alg,
        maxiters=500,
    )

    # OU analytic solution
    σ² = 0.5      # stationary variance = 1/2 <- # $Var_{\infty} = \frac{\beta^2}{2|\alpha|}$
    analytic_sol_func(x, t) = pdf(Normal(u0 * exp(-t), sqrt(σ² * (1 - exp(-2t)))), x) # mean μ and variance σ^2
    xs = collect(x_0:dx:x_end)

    # test at 0.1, not 0.0 ∵ analytic sol goes to inf (dirac delta func)
    ts = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    u_real = [[analytic_sol_func(x, t) for x in xs] for t in ts]
    u_predict = [[first(phi([x, t], sol_OU.u)) for x in xs] for t in ts]  # NeuralPDE predictions

    # MSE across all x.
    diff = u_real .- u_predict
    @test mean(vcat([abs2.(diff_i) for diff_i in diff]...)) < 0.01

    # using Plots
    # plotly()
    # plots_got = []
    # for i in 1:length(ts)
    #     plot(xs, u_real[i], label="analytic t=$(ts[i])")
    #     push!(plots_got, plot!(xs, u_predict[i], label="predict t=$(ts[i])"))
    # end
    # plot(plots_got..., legend=:outerbottomright)
end

@testitem "GBM SDE" tags = [:nnsde2] begin
    using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Optimisers
    using OrdinaryDiffEq, Random, Distributions, Integrals, Cubature
    using DifferentialEquations, LineSearches
    using ModelingToolkit: infimum, supremum
    using OptimizationOptimJL: BFGS
    Random.seed!(100)

    μ = 0.2
    σ = 0.3
    f(x, p, t) = μ * x
    g(x, p, t) = σ * x
    u0 = 1.0
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u0, tspan)

    # Neural network
    inn = 20
    chain = Lux.Chain(Dense(2, inn, Lux.tanh),
        Dense(inn, inn, Lux.tanh),
        Dense(inn, 1, Lux.logcosh
        )) |> f64

    # problem setting - (results depend on x's assumed range)

    dx = 0.01
    x_0 = 0.0
    x_end = 3.0
    σ_var_bc = 0.05
    alg = SDEPINN(
        chain=chain,
        optimalg=BFGS(),
        norm_loss_alg=HCubatureJL(),
        x_0=x_0,
        x_end=x_end,

        # pdf(LogNormal(log(X₀), σ_var_bc), x)  # initial PDF
        # for gbm normal X0 disti also gives good results with absorbing_bc.
        distrib=LogNormal(log(u0), σ_var_bc)
    )

    sol_GBM, phi = solve(
        prob,
        alg,
        maxiters=500
    )

    analytic_sol_func(x, t) = pdf(LogNormal(log(u0) + (μ - 0.5 * σ^2) * t, sqrt(t) * σ), x)
    xs = collect(x_0:dx:x_end)

    # test at 0.1, not 0.0 ∵ analytic sol goes to inf (dirac delta func)
    ts = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    u_real = [[analytic_sol_func(x, t) for x in xs] for t in ts]
    u_predict = [[first(phi([x, t], sol_GBM.u)) for x in xs] for t in ts]

    # MSE across all x.
    diff = u_real .- u_predict
    @test mean(vcat([abs2.(diff_i) for diff_i in diff]...)) < 0.01

    # Compare with analytic GBM solution
    # using Plots
    # plotly()
    # plots_got = []
    # for i in 1:length(ts)
    #     plot(xs, u_real[i], label="analytic t=$(ts[i])")
    #     push!(plots_got, plot!(xs, u_predict[i], label="predict t=$(ts[i])"))
    # end
    # plot(plots_got..., legend=:outerbottomright)
end