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

@testitem "Test-2" tags=[:nnsde] begin
    using OrdinaryDiffEq, Random, Lux, Optimisers, DiffEqNoiseProcess, Distributions
    using OptimizationOptimJL: BFGS

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
    abstol = 1e-10
    autodiff = false
    kwargs = (; verbose = true, dt = dt, abstol, maxiters = 300)
    opt = BFGS()
    numensemble = 1000

    sol_1 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true);
        kwargs...)
    sol_2 = solve(
        prob, NNSDE(
            luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 10, batch = true);
        kwargs...)

    ts = sol_1.timepoints
    u1 = sol_1.mean_fit
    u2 = sol_2.mean_fit

    analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
    function W_kkl(t, z1, z2, z3)
        √2 * (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
         z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
         z3 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π))
    end
    truncated_sol(u0, t, z1, z2, z3) = u0 *
                                       exp((α - β^2 / 2) * t + β * W_kkl(t, z1, z2, z3))

    # sub_sampling for training improves stability of results across different num_samples choices
    # for higher num_samples, we almost always get a good result for > sub_sample case.
    # as we reduce num_samples, < and > sub_sample cases start performing better (mean against analytic_sol and truncated_sol)
    # but the Rate of improvements of less < subsample case seems to be higher as we reduce num_samples.

    num_samples = 2000
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
            predicted_solution_samples_2[i, j] = u₀ .+
                                                 (ts[i] - ts[1]) .*
                                                 sol_2.solution.interp.phi(
                phi_inputs[j][:, i], sol_2.solution.interp.θ)

            truncated_solution_samples[i, j] = truncated_sol(
                u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j])
        end
    end

    mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
    mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
    mean_predicted_solution_1 = mean(predicted_solution_samples_1, dims = 2)
    mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

    error_1 = sum(abs2, mean_analytic_solution .- u1)
    error_2 = sum(abs2, mean_analytic_solution .- u2)
    @test error_1 > error_2

    MSE_1 = mean(abs2.(mean_analytic_solution .- u1))
    MSE_2 = mean(abs2.(mean_analytic_solution .- u2))
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
    @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 1e-1
    @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 5e-2
end