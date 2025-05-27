@testitem "Test-1" tags=[:nnsde] begin
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
    dim = 1 + 3
    luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))

    @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
        autodiff in [false, true]

        if autodiff
            @test_throws ArgumentError solve(
                prob, NNSDE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0)
            continue
        end

        @testset for (dt, abstol) in [(1 / 20.0f0, 1e-10), (nothing, 1e-6)]
            kwargs = (; verbose = false, dt, abstol, maxiters = 200)
            sol = solve(prob, NNSDE(luxchain, opt; autodiff); kwargs...)
        end
    end
end

@testitem "Test - GBM SDE" tags=[:nnsde] begin
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
    dim = 1 + 3
    luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))

    dt = 1 / 50.0f0
    abstol = 1e-6
    autodiff = false
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 300)
    opt = BFGS()
    numensemble = 2000

    sol_2 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 10, batch = true);
        kwargs...)

    sol_1 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true);
        kwargs...)

    # sol_1 and sol_2 have same timespan
    ts = sol_1.timepoints
    u1 = sol_1.strong_sol
    u2 = sol_2.strong_sol

    analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
    function W_kkl(t, z1, z2, z3)
        √2 * (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
         z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
         z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π))
    end
    truncated_sol(u0, t, z1, z2, z3) = u0 *
                                       exp((α - β^2 / 2) * t + β * W_kkl(t, z1, z2, z3))

    num_samples = 3000
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
    phi_inputs = [hcat([vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)]...)
                  for i in 1:num_samples]

    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_1 = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_2 = Array{Float64}(undef, num_time_steps, num_samples)

    for j in 1:num_samples
        for i in 1:num_time_steps
            # for each sample, pass each timepoints and get output
            analytic_solution_samples[i, j] = analytic_sol(u₀, 0, ts[i], W_samples[i, j])

            predicted_solution_samples_1[i, j] = sol_1.solution.interp.phi(
                phi_inputs[j][:, i], sol_1.solution.interp.θ)
            predicted_solution_samples_2[i, j] = sol_2.solution.interp.phi(
                phi_inputs[j][:, i], sol_2.solution.interp.θ)

            truncated_solution_samples[i, j] = truncated_sol(
                u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j])
        end
    end

    # strong solution tests
    strong_analytic_solution = [Particles(analytic_solution_samples[i, :])
                                for i in eachindex(ts)]
    strong_truncated_solution = [Particles(truncated_solution_samples[i, :])
                                 for i in eachindex(ts)]
    strong_predicted_solution_1 = [Particles(predicted_solution_samples_1[i, :])
                                   for i in eachindex(ts)]
    strong_predicted_solution_2 = [Particles(predicted_solution_samples_2[i, :])
                                   for i in eachindex(ts)]

    error_1 = sum(abs2, strong_analytic_solution .- strong_predicted_solution_1)
    error_2 = sum(abs2, strong_analytic_solution .- strong_predicted_solution_2)
    @test pmean(error_1) > pmean(error_2)

    @test pmean(sum(abs2.(strong_predicted_solution_1 .- strong_truncated_solution))) >
          pmean(sum(abs2.(strong_predicted_solution_2 .- strong_truncated_solution)))

    # weak solution tests
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
    @test MSE_2 < 5e-2

    error_1 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_1)
    error_2 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_2)
    @test error_1 > error_2

    MSE_1 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_1))
    MSE_2 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2))
    @test MSE_2 < MSE_1
    @test MSE_2 < 5e-2

    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) >
          mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution))
    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 3e-1
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 4e-2
end

# Equation 65 from https://arxiv.org/abs/1804.04344
@testitem "Test-3 Additive Noise Test Equation" tags=[:nnsde] begin
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
    dim = 1 + 6
    luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, tanh), Dense(16, 16, σ), Dense(16, 1))

    dt = 1 / 50.0f0
    abstol = 1e-7
    autodiff = false
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 300)
    opt = BFGS()
    numensemble = 2000

    sol_1 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true);
        kwargs...)

    sol_2 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 10, batch = true);
        kwargs...)

    # sol_1 and sol_2 have same timespan
    ts = sol_1.timepoints
    u1 = sol_1.strong_sol
    u2 = sol_2.strong_sol

    analytic_sol(u0, p, t, W) = (u0 / sqrt(1 + t)) + (β * (t + α * W) / sqrt(1 + t))
    function W_kkl(t, z1, z2, z3, z4, z5, z6)
        √2 * ((z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π)) +
         (z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π)) +
         (z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π)) +
         (z4 * sin((4 - 1 / 2) * π * t) / ((4 - 1 / 2) * π)) +
         (z5 * sin((5 - 1 / 2) * π * t) / ((5 - 1 / 2) * π)) +
         (z6 * sin((6 - 1 / 2) * π * t) / ((6 - 1 / 2) * π)))
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
        z1_samples, z2_samples, z3_samples, z4_samples, z5_samples, z6_samples)'
    phi_inputs = [hcat([vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)]...)
                  for i in 1:num_samples]

    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_1 = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_2 = Array{Float64}(undef, num_time_steps, num_samples)

    for j in 1:num_samples
        for i in 1:num_time_steps
            # for each sample, pass each timepoints and get output
            analytic_solution_samples[i, j] = analytic_sol(u₀, 0, ts[i], W_samples[i, j])

            predicted_solution_samples_1[i, j] = sol_1.solution.interp.phi(
                phi_inputs[j][:, i], sol_1.solution.interp.θ)
            predicted_solution_samples_2[i, j] = sol_2.solution.interp.phi(
                phi_inputs[j][:, i], sol_2.solution.interp.θ)

            truncated_solution_samples[i, j] = truncated_sol(
                u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j],
                z4_samples[j], z5_samples[j], z6_samples[j])
        end
    end

    # strong solution tests
    strong_analytic_solution = [Particles(analytic_solution_samples[i, :])
                                for i in eachindex(ts)]
    strong_truncated_solution = [Particles(truncated_solution_samples[i, :])
                                 for i in eachindex(ts)]
    strong_predicted_solution_1 = [Particles(predicted_solution_samples_1[i, :])
                                   for i in eachindex(ts)]
    strong_predicted_solution_2 = [Particles(predicted_solution_samples_2[i, :])
                                   for i in eachindex(ts)]

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
    @test MSE_1 < 5e-5
    @test MSE_2 < 3e-5

    error_1 = sum(abs2, mean_truncated_solution .- mean_predicted_solution_1)
    error_2 = sum(abs2, mean_truncated_solution .- mean_predicted_solution_2)
    @test error_1 > error_2
    @test error_2 < 3e-3

    MSE_1 = mean(abs2.(mean_truncated_solution .- mean_predicted_solution_1))
    MSE_2 = mean(abs2.(mean_truncated_solution .- mean_predicted_solution_2))
    @test MSE_2 < MSE_1
    @test MSE_2 < 3e-5
end

@testitem "SDE Parameter Estimation" tags=[:nnsde] begin
    using NeuralPDE
    using OrdinaryDiffEq, Random, Lux, Optimisers, DiffEqNoiseProcess, Distributions
    using OptimizationOptimJL: BFGS
    Random.seed!(100)

    α = 1.2
    β = 1.1
    param_values = α
    u₀ = 0.5

    f(u, p, t) = p * u
    g(u, p, t) = β * u
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u₀, tspan, param_values)
    dim = 1 + 3
    dt = 1 / 100.0f0
    luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))

    analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
    function W_kkl(t, z1, z2, z3)
        √2 * (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
         z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
         z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π))
    end

    num_samples = 10000
    num_time_steps = dt
    z1_samples = rand(Normal(0, 1), num_samples)
    z2_samples = rand(Normal(0, 1), num_samples)
    z3_samples = rand(Normal(0, 1), num_samples)

    ts = collect(tspan[1]:dt:tspan[2])
    num_time_steps = size(ts)[1]
    W_samples = Array{Float64}(undef, num_time_steps, num_samples)

    for i in 1:num_samples
        W = WienerProcess(0.0, 0.0)
        probtemp = NoiseProblem(W, (0.0, 1.0))
        Np_sol = solve(probtemp; dt = dt)
        W_samples[:, i] = Np_sol.u
    end
    temp_rands = hcat(z1_samples, z2_samples, z3_samples)'
    phi_inputs = [hcat([vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)]...)
                  for i in 1:num_samples]

    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for j in 1:num_samples
        for i in 1:num_time_steps
            # for each sample, pass each timepoints and get output
            analytic_solution_samples[i, j] = analytic_sol(
                u₀, param_values, ts[i], W_samples[i, j])
        end
    end
    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    plot(mean_analytic_solution)

    function additional_loss(phi, θ)
        return sum(abs2,
            [mean([phi(phi_inputs[j][:, i], θ) for j in 1:10]) for i in 1:101] .-
            mean_analytic_solution) / 101
    end

    abstol = 1e-10
    autodiff = false
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 300)
    opt = BFGS()
    numensemble = 2000

    alg = NNSDE(luxchain, opt; autodiff, numensemble = numensemble,
        sub_batch = 10, batch = true, param_estim = true, additional_loss)

    sol_1 = solve(prob, alg; kwargs...)

    # sol_1 and sol_2 have same timespan
    ts = sol_1.timepoints
    u1 = sol_1.mean_fit
    u2 = sol_2.mean_fit

    analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
    function W_kkl(t, z1, z2, z3)
        √2 * (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
         z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
         z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π))
    end
    truncated_sol(u0, t, z1, z2, z3) = u0 *
                                       exp((α - β^2 / 2) * t + β * W_kkl(t, z1, z2, z3))

    num_samples = 10
    num_time_steps = dt
    z1_samples = rand(Normal(0, 1), num_samples)
    z2_samples = rand(Normal(0, 1), num_samples)
    z3_samples = rand(Normal(0, 1), num_samples)

    num_time_steps = size(ts)[1]
    W_samples = Array{Float64}(undef, num_time_steps, num_samples)
    for i in 1:num_samples
        W = WienerProcess(0.0, 0.0)
        probtemp = NoiseProblem(W, (0.0, 1.0))
        Np_sol = solve(probtemp; dt = dt)
        W_samples[:, i] = Np_sol.u
    end

    temp_rands = hcat(z1_samples, z2_samples, z3_samples)'
    phi_inputs = [hcat([vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)]...)
                  for i in 1:num_samples]

    analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_1 = Array{Float64}(undef, num_time_steps, num_samples)
    predicted_solution_samples_2 = Array{Float64}(undef, num_time_steps, num_samples)

    for j in 1:num_samples
        for i in 1:num_time_steps
            # for each sample, pass each timepoints and get output
            analytic_solution_samples[i, j] = analytic_sol(u₀, 0, ts[i], W_samples[i, j])

            predicted_solution_samples_1[i, j] = u₀ .+
                                                 (ts[i] - ts[1]) .*
                                                 sol_1.solution.interp.phi(
                phi_inputs[j][:, i], sol_1.solution.interp.θ)
            # predicted_solution_samples_2[i, j] = u₀ .+
            #  (ts[i] - ts[1]) .*
            #  sol_2.solution.interp.phi(
            # phi_inputs[j][:, i], sol_2.solution.interp.θ)

            truncated_solution_samples[i, j] = truncated_sol(
                u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j])
        end
    end

    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
    mean_predicted_solution_1 = mean(predicted_solution_samples_1, dims = 2)
    mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

    # using Plots
    # plotly()
    # plot(mean_analytic_solution)
    # plot!(u1)
    # plot!(mean_predicted_solution_1)
    # sol_1.solution.interp.θ[:p]

    # testing over different Z_i sample sizes
    error_1 = sum(abs2, mean_analytic_solution .- u1)
    error_2 = sum(abs2, mean_analytic_solution .- u2)
    @test error_1 > error_2

    MSE_1 = mean(abs2.(mean_analytic_solution .- u1))
    MSE_2 = mean(abs2.(mean_analytic_solution .- u2))
    @test MSE_2 < MSE_1
    @test MSE_2 < 1e-2

    error_1 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_1)
    error_2 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_2)
    @test error_1 > error_2

    MSE_1 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_1))
    MSE_2 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2))
    @test MSE_2 < MSE_1
    @test MSE_2 < 1e-2

    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) >
          mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution))
    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 5e-2
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 2e-2
end