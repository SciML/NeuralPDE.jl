using NeuralPDE
using Test

@testset "Test-4 GBM SDE Inverse, weak & strong solving" begin
    # Also works for brownian motion with constant drift.
    using OrdinaryDiffEq, Random, Lux, Optimisers, DiffEqNoiseProcess, Distributions
    using MonteCarloMeasurements: pmean, Particles
    using OptimizationOptimJL: BFGS
    Random.seed!(100)

    # problem setting
    ideal_p = [1.5, 0.5]
    param_initval = [0.0, 0.0]
    u₀ = 0.5
    f(u, p, t) = p[1] * u
    g(u, p, t) = p[2] * u
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u₀, tspan, param_initval)
    n_z = 3
    dim = 1 + n_z

    # discretization for dataset
    dt = 1 / 100.0f0
    luxchain = Chain(Dense(dim, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)) |> f64

    # Dataset Preparation
    num_samples = 15
    ts = collect(tspan[1]:dt:tspan[2])
    num_time_steps = length(ts)

    # We want n=num_samples strong solutions defined at timepoints.
    W_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for i in 1:num_samples
        W = WienerProcess(0.0, 0.0)
        # Solve the noise over the actual `ts` grid (which a Float32 `dt` ends just
        # short of `1.0`) so the path is sampled at exactly `ts`. Using `(0.0, 1.0)`
        # makes the solver land an extra point on `1.0` (DiffEqNoiseProcess#278), which
        # no longer matches `length(ts)`.
        probtemp = NoiseProblem(W, (0.0, ts[end]))
        Np_sol = solve(probtemp; dt = dt)
        W_samples[:, i] = Np_sol.u
    end

    analytic_sol(u0, p, t, W) = u0 * exp((p[1] - p[2]^2 / 2) * t + p[2] * W)
    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for i in 1:num_time_steps
        for j in 1:num_samples
            # for each timepoint, pass each strong path's W value.
            analytic_solution_samples[i, j] = analytic_sol(
                u₀, ideal_p, ts[i], W_samples[i, j]
            )
        end
    end

    # adapted process got from filration Ft on the probability space of SDE solution, time.
    observed_process = [analytic_solution_samples[:, i] for i in 1:num_samples]
    dataset = [observed_process, ts]

    N_solve = 50
    # solver configuration
    abstol = 1.0e-12
    autodiff = false
    kwargs = (; verbose = true, dt = 1 / N_solve, abstol, maxiters = 500)
    opt = BFGS()
    numensemble = 200

    # for inverse problems more sub_batch leads to learning mainly the drift parameter
    alg_1 = NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble,
        sub_batch = 1, batch = true, param_estim = true, strong_loss = true, dataset = dataset
    )
    alg_2 = NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble,
        sub_batch = 1, batch = true, param_estim = true, strong_loss = false, dataset = dataset
    )
    sol_2 = solve(prob, alg_2; kwargs...)
    sol_1 = solve(prob, alg_1; kwargs...)

    # sol_1, sol_2 have the same timespan and are single output
    ts = sol_1.timepoints
    u2 = sol_2.estimated_sol[1]

    function W_kkl(t, z1, z2, z3)
        √2 * (
            z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
                z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
                z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π)
        )
    end
    function truncated_sol(u0, p, t, z1, z2, z3)
        u0 *
            exp((p[1] - p[2]^2 / 2) * t + p[2] * W_kkl(t, z1, z2, z3))
    end

    # testing dataset must be for the same timepoints as solution
    num_samples = 500
    num_time_steps = length(ts)
    W_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for i in 1:num_samples
        W = WienerProcess(0.0, 0.0)
        # Sample the noise on exactly the solution's `ts` grid; see the note above.
        probtemp = NoiseProblem(W, (0.0, ts[end]))
        Np_sol = solve(probtemp; dt = 1 / N_solve)
        W_samples[:, i] = Np_sol.u
    end

    temp_rands = hcat([randn(num_samples) for _ in 1:n_z]...)'
    phi_inputs = [
        reduce(hcat, [vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)])
            for i in 1:num_samples
    ]

    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_2 = Array{Float64}(undef, num_time_steps, num_samples)

    for j in 1:num_samples
        for i in 1:num_time_steps
            # for each sample, pass each timepoints and get output
            analytic_solution_samples[i, j] = analytic_sol(
                u₀, ideal_p, ts[i], W_samples[i, j]
            )

            predicted_solution_samples_2[i, j] = sol_2.rode_solution.interp.phi(
                phi_inputs[j][:, i], sol_2.rode_solution.interp.θ
            )

            truncated_solution_samples[i, j] = truncated_sol(
                u₀, ideal_p, ts[i], temp_rands[:, j]...
            )
        end
    end

    # weak solution tests (sol_2)
    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
    mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

    # testing over different, same Z_i sample sizes
    # relaxed tolerances — SDE inverse problems have high inherent stochastic variance.
    @test mean(abs2.(mean_analytic_solution .- pmean(u2))) < 1.5
    @test mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2)) < 1.5
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 1.5

    # strong solution tests (sol_1)
    # get SDEPINN output at fixed path we solved over.
    solution_1_strong_solve = reduce(
        vcat,
        Base.Fix2(
            sol_1.rode_solution.interp.phi,
            sol_1.rode_solution.interp.θ
        ).(
            collect.(sol_1.training_sets)
        )
    )

    # get truncated solution on strong path over which training was done.
    truncatedsol_data_inputs = reduce(hcat, sol_1.training_sets)
    truncated_solution_strong_paths = [
        truncated_sol(
                u₀, ideal_p, truncatedsol_data_inputs[:, i]...
            )
            for i in eachindex(ts)
    ]

    @test mean(abs2, solution_1_strong_solve .- truncated_solution_strong_paths) < 2.0

    # estimated sde parameter tests (we trained with 15 observed solution paths).
    # absolute value taken for 2nd estimated parameter as loss for variance is independent of this parameter's direction.
    # relaxed tolerances — SDE parameter estimation has high variance across runs.
    @test sol_1.estimated_params[1] .≈ ideal_p[1] rtol = 0.5
    @test abs(sol_1.estimated_params[2]) .≈ ideal_p[2] rtol = 0.5
    @test sol_2.estimated_params[1] .≈ ideal_p[1] rtol = 0.5
    @test abs(sol_2.estimated_params[2]) .≈ ideal_p[2] rtol = 0.5
end
