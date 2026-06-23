using NeuralPDE
using Test

@testset "Test-2 GBM SDE" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, DiffEqNoiseProcess, Distributions
    using OptimizationOptimJL: BFGS
    using MonteCarloMeasurements: Particles, pmean
    Random.seed!(100)

    α = 1.2
    β = 1.1
    u₀ = 0.5
    f(u, p, t) = α * u
    g(u, p, t) = β * u
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u₀, tspan)
    n_z = 3
    dim = 1 + n_z
    luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1)) |> f64

    dt = 1 / 50.0f0
    abstol = 1.0e-12
    autodiff = false
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 500)
    opt = BFGS()
    numensemble = 500

    sol_2 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 10, batch = true
        );
        kwargs...
    )

    sol_1 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true
        );
        kwargs...
    )

    # sol_1 and sol_2 have same timespan
    ts = sol_1.timepoints
    u1 = sol_1.estimated_sol[1]
    u2 = sol_2.estimated_sol[1]

    analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
    function W_kkl(t, z1, z2, z3)
        √2 * (
            z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
                z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
                z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π)
        )
    end
    truncated_sol(
        u0, t, z1, z2, z3
    ) = u0 *
        exp((α - β^2 / 2) * t + β * W_kkl(t, z1, z2, z3))

    num_samples = 2000
    num_time_steps = dt
    z1_samples = rand(Normal(0, 1), num_samples)
    z2_samples = rand(Normal(0, 1), num_samples)
    z3_samples = rand(Normal(0, 1), num_samples)

    num_time_steps = length(ts)
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

    temp_rands = hcat(z1_samples, z2_samples, z3_samples)'
    phi_inputs = [
        hcat([vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)]...)
            for i in 1:num_samples
    ]

    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_1 = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_2 = Array{Float64}(undef, num_time_steps, num_samples)

    for j in 1:num_samples
        for i in 1:num_time_steps
            # for each sample, pass each timepoints and get output
            analytic_solution_samples[i, j] = analytic_sol(u₀, 0, ts[i], W_samples[i, j])

            predicted_solution_samples_1[i, j] = sol_1.rode_solution.interp.phi(
                phi_inputs[j][:, i], sol_1.rode_solution.interp.θ
            )
            predicted_solution_samples_2[i, j] = sol_2.rode_solution.interp.phi(
                phi_inputs[j][:, i], sol_2.rode_solution.interp.θ
            )

            truncated_solution_samples[i, j] = truncated_sol(
                u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j]
            )
        end
    end

    # strong ensemble solution tests
    strong_analytic_solution = [
        Particles(analytic_solution_samples[i, :])
            for i in eachindex(ts)
    ]
    strong_truncated_solution = [
        Particles(truncated_solution_samples[i, :])
            for i in eachindex(ts)
    ]
    strong_predicted_solution_1 = [
        Particles(predicted_solution_samples_1[i, :])
            for i in eachindex(ts)
    ]
    strong_predicted_solution_2 = [
        Particles(predicted_solution_samples_2[i, :])
            for i in eachindex(ts)
    ]

    error_1 = sum(abs2, strong_analytic_solution .- strong_predicted_solution_1)
    error_2 = sum(abs2, strong_analytic_solution .- strong_predicted_solution_2)
    @test pmean(error_1) > pmean(error_2) - 10.0

    @test pmean(sum(abs2.(strong_predicted_solution_1 .- strong_truncated_solution))) + 10.0 >
        pmean(sum(abs2.(strong_predicted_solution_2 .- strong_truncated_solution)))

    # weak ensemble solution tests
    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
    mean_predicted_solution_1 = mean(predicted_solution_samples_1, dims = 2)
    mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

    # testing over different Z_i sample sizes
    error_1 = sum(abs2, mean_analytic_solution .- pmean(u1))
    error_2 = sum(abs2, mean_analytic_solution .- pmean(u2))
    @test error_1 > error_2 - 4.0

    MSE_1 = mean(abs2.(mean_analytic_solution .- pmean(u1)))
    MSE_2 = mean(abs2.(mean_analytic_solution .- pmean(u2)))
    @test MSE_2 < MSE_1 + 0.1
    @test MSE_2 < 2.0e-1

    error_1 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_1)
    error_2 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_2)
    @test error_1 > error_2 - 4.0

    MSE_1 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_1))
    MSE_2 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2))
    @test MSE_2 < MSE_1 + 0.1
    @test MSE_2 < 2.0e-1

    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) + 0.1 >
        mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution))
    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 6.0e-1
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 2.0e-1
end

# Equation 65 from https://arxiv.org/abs/1804.04344
