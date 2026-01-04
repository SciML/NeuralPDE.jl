@testitem "Test-1 solve & autodiff" tags = [:nnsde] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    using OptimizationOptimJL: BFGS
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

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
            autodiff in [false, true]

        if autodiff
            @test_throws ArgumentError solve(
                prob, NNSDE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0
            )
            continue
        end

        @testset for (dt, abstol) in [(1 / 20.0f0, 1.0e-10), (nothing, 1.0e-6)]
            kwargs = (; verbose = false, dt, abstol, maxiters = 200)
            sol = solve(prob, NNSDE(luxchain, opt; autodiff); kwargs...)
        end
    end
end

@testitem "Test-2 GBM SDE" tags = [:nnsde] begin
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
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 400)
    opt = BFGS()
    numensemble = 1000

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
        probtemp = NoiseProblem(W, (0.0, 1.0))
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
    @test pmean(error_1) > pmean(error_2)

    @test pmean(sum(abs2.(strong_predicted_solution_1 .- strong_truncated_solution))) >
        pmean(sum(abs2.(strong_predicted_solution_2 .- strong_truncated_solution)))

    # weak ensemble solution tests
    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
    mean_predicted_solution_1 = mean(predicted_solution_samples_1, dims = 2)
    mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

    # testing over different Z_i sample sizes
    error_1 = sum(abs2, mean_analytic_solution .- pmean(u1))
    error_2 = sum(abs2, mean_analytic_solution .- pmean(u2))
    @test error_1 > error_2

    MSE_1 = mean(abs2.(mean_analytic_solution .- pmean(u1)))
    MSE_2 = mean(abs2.(mean_analytic_solution .- pmean(u2)))
    @test MSE_2 < MSE_1
    @test MSE_2 < 5.0e-2

    error_1 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_1)
    error_2 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_2)
    @test error_1 > error_2

    MSE_1 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_1))
    MSE_2 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2))
    @test MSE_2 < MSE_1
    @test MSE_2 < 5.0e-2

    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) >
        mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution))
    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 6.0e-1
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 4.0e-2
end

# Equation 65 from https://arxiv.org/abs/1804.04344
@testitem "Test-3 Additive Noise Test Equation" tags = [:nnsde] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, DiffEqNoiseProcess, Distributions
    using OptimizationOptimJL: BFGS
    using MonteCarloMeasurements: Particles, pmean
    Random.seed!(100)

    α = 0.1
    β = 0.05
    u₀ = 0.5
    f(u, p, t) = (β / sqrt(1 + t)) - (u[1] / ((1 + t) * 2))
    g(u, p, t) = β * α / sqrt(1 + t)
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u₀, tspan)
    n_z = 6
    dim = 1 + n_z
    luxchain = Chain(
        Dense(dim, 16, σ), Dense(16, 16, tanh), Dense(16, 16, σ), Dense(16, 1)
    ) |> f64

    dt = 1 / 50.0f0
    abstol = 1.0e-7
    autodiff = false
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 300)
    opt = BFGS()
    numensemble = 2000

    sol_1 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true
        );
        kwargs...
    )

    sol_2 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 10, batch = true
        );
        kwargs...
    )

    # sol_1, sol_2 have the same timespan and are single output.
    ts = sol_1.timepoints
    u1 = sol_1.estimated_sol[1]
    u2 = sol_2.estimated_sol[1]

    analytic_sol(u0, p, t, W) = (u0 / sqrt(1 + t)) + (β * (t + α * W) / sqrt(1 + t))
    function W_kkl(t, z1, z2, z3, z4, z5, z6)
        √2 * (
            (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π)) +
                (z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π)) +
                (z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π)) +
                (z4 * sin((4 - 1 / 2) * π * t) / ((4 - 1 / 2) * π)) +
                (z5 * sin((5 - 1 / 2) * π * t) / ((5 - 1 / 2) * π)) +
                (z6 * sin((6 - 1 / 2) * π * t) / ((6 - 1 / 2) * π))
        )
    end
    function truncated_sol(u0, t, z1, z2, z3, z4, z5, z6)
        (u0 / sqrt(1 + t)) + (β * (t + α * W_kkl(t, z1, z2, z3, z4, z5, z6)) / sqrt(1 + t))
    end

    num_samples = 3000
    num_time_steps = dt
    z1_samples = rand(Normal(0, 1), num_samples)
    z2_samples = rand(Normal(0, 1), num_samples)
    z3_samples = rand(Normal(0, 1), num_samples)
    z4_samples = rand(Normal(0, 1), num_samples)
    z5_samples = rand(Normal(0, 1), num_samples)
    z6_samples = rand(Normal(0, 1), num_samples)

    num_time_steps = length(ts)
    W_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for i in 1:num_samples
        W = WienerProcess(0.0, 1.0)
        probtemp = NoiseProblem(W, (0.0, 1.0))
        Np_sol = solve(probtemp; dt = dt)
        W_samples[:, i] = Np_sol.u
    end

    temp_rands = hcat(
        z1_samples, z2_samples, z3_samples, z4_samples, z5_samples, z6_samples
    )'
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
                u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j],
                z4_samples[j], z5_samples[j], z6_samples[j]
            )
        end
    end

    # strong solution tests
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
    @test pmean(error_1) > pmean(error_2)

    error1 = sum(abs2.(strong_predicted_solution_1 .- strong_truncated_solution))
    error2 = sum(abs2.(strong_predicted_solution_2 .- strong_truncated_solution))
    @test pmean(error1) > pmean(error2)

    # weak solution tests
    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
    mean_predicted_solution_1 = mean(predicted_solution_samples_1, dims = 2)
    mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

    # testing over different Z_i sample sizes
    MSE_1 = mean(abs2.(mean_analytic_solution .- pmean(u1)))
    MSE_2 = mean(abs2.(mean_analytic_solution .- pmean(u2)))
    @test MSE_1 < 1.0e-4
    @test MSE_2 < 8.0e-5

    error_1 = sum(abs2, mean_truncated_solution .- mean_predicted_solution_1)
    error_2 = sum(abs2, mean_truncated_solution .- mean_predicted_solution_2)
    @test error_1 > error_2
    @test error_2 < 5.0e-3

    MSE_1 = mean(abs2.(mean_truncated_solution .- mean_predicted_solution_1))
    MSE_2 = mean(abs2.(mean_truncated_solution .- mean_predicted_solution_2))
    @test MSE_2 < MSE_1
    @test MSE_2 < 8.0e-5
end

@testitem "Test-4 GBM SDE Inverse, weak & strong solving" tags = [:nnsde] begin
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
        probtemp = NoiseProblem(W, (0.0, 1.0))
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

    # solver configuration
    abstol = 1.0e-12
    autodiff = false
    kwargs = (; verbose = true, dt = 1 / 50.0f0, abstol, maxiters = 700)
    opt = BFGS()
    numensemble = 100

    # for inverse problems more sub_batch leads to learning mainly the drift parameter
    alg_1 = NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble,
        sub_batch = 1, batch = true, param_estim = true, strong_loss = true, dataset = dataset
    )
    alg_2 = NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble,
        sub_batch = 1, batch = true, param_estim = true, strong_loss = false, dataset = dataset
    )
    sol_1 = solve(prob, alg_1; kwargs...)
    sol_2 = solve(prob, alg_2; kwargs...)

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
    num_samples = 200
    num_time_steps = length(ts)
    W_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for i in 1:num_samples
        W = WienerProcess(0.0, 0.0)
        probtemp = NoiseProblem(W, (0.0, 1.0))
        Np_sol = solve(probtemp; dt = kwargs.dt)
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
    # relaxed tolerances for Julia pre and v1 in the below tests.
    # All the below Tests pass for lts-Julia v1.10.10 with tolerances as < 5e-2.
    @test mean(abs2.(mean_analytic_solution .- pmean(u2))) < 0.16
    @test mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2)) < 0.22
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 0.21

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

    @test mean(abs2, solution_1_strong_solve .- truncated_solution_strong_paths) < 5.0e-2

    # estimated sde parameter tests (we trained with 15 observed solution paths).
    # absolute value taken for 2nd estimated parameter as loss for variance is independent of this parameter's direction.
    @test sol_1.estimated_params[1] .≈ ideal_p[1] rtol = 2.0e-1
    @test abs(sol_1.estimated_params[2]) .≈ ideal_p[2] rtol = 8.0e-2
    @test sol_2.estimated_params[1] .≈ ideal_p[1] rtol = 2.0e-1
    @test abs(sol_2.estimated_params[2]) .≈ ideal_p[2] rtol = 8.0e-2
end
