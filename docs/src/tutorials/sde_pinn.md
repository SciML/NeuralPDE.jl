# # GBM SDE
# using OrdinaryDiffEq, Random, Lux, Optimisers, DiffEqNoiseProcess, Distributions
# using OptimizationOptimJL: BFGS
# using MonteCarloMeasurements: Particles, pmean
# Random.seed!(100)
# using NeuralPDE, Test
# α = 1.2
# β = 1.1
# u₀ = 0.5
# f(u, p, t) = α * u
# g(u, p, t) = β * u
# tspan = (0.0, 1.0)
# prob = SDEProblem(f, g, u₀, tspan)
# dim = 1 + 3
# luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))

# dt = nothing
# #  1 / 50.0f0
# abstol = 1e-6
# autodiff = false
# kwargs = (; verbose = true, dt = dt, abstol, maxiters = 300)
# opt = BFGS()
# numensemble = 2000

# sol_2 = solve(
#     prob, NNSDE(
#         luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 10, batch = true);
#     kwargs...)

# sol_1 = solve(
#     prob, NNSDE(
#         luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true);
#     kwargs...)

# # sol_1 and sol_2 have same timespan
# ts = sol_1.timepoints
# u1 = sol_1.strong_sol
# u2 = sol_2.strong_sol

# analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
# function W_kkl(t, z1, z2, z3)
#     √2 * (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
#      z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
#      z3 * sin((3 - 1 / 2) * π * t) / ((3 - 1 / 2) * π))
# end
# function truncated_sol(u0, t, z1, z2, z3)
#     u0 *
#     exp((α - β^2 / 2) * t + β * W_kkl(t, z1, z2, z3))
# end

# num_samples = 3000
# num_time_steps = dt
# z1_samples = rand(Normal(0, 1), num_samples)
# z2_samples = rand(Normal(0, 1), num_samples)
# z3_samples = rand(Normal(0, 1), num_samples)

# num_time_steps = length(ts)
# W_samples = Array{Float64}(undef, num_time_steps, num_samples)
# for i in 1:num_samples
#     W = WienerProcess(0.0, 0.0)
#     probtemp = NoiseProblem(W, (0.0, 1.0))
#     Np_sol = solve(probtemp; dt = dt)
#     W_samples[:, i] = Np_sol.u
# end

# temp_rands = hcat(
#     z1_samples, z2_samples, z3_samples)'
# phi_inputs = [hcat([vcat(ts[j], temp_rands[:, i]) for j in eachindex(ts)]...)
#               for i in 1:num_samples]

# analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
# truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
# predicted_solution_samples_1 = Array{Float64}(undef, num_time_steps, num_samples)
# predicted_solution_samples_2 = Array{Float64}(undef, num_time_steps, num_samples)

# for j in 1:num_samples
#     for i in 1:num_time_steps
#         # for each sample, pass each timepoints and get output
#         analytic_solution_samples[i, j] = analytic_sol(u₀, 0, ts[i], W_samples[i, j])

#         predicted_solution_samples_1[i, j] = sol_1.solution.interp.phi(
#             phi_inputs[j][:, i], sol_1.solution.interp.θ)
#         predicted_solution_samples_2[i, j] = sol_2.solution.interp.phi(
#             phi_inputs[j][:, i], sol_2.solution.interp.θ)

#         truncated_solution_samples[i, j] = truncated_sol(
#             u₀, ts[i], z1_samples[j], z2_samples[j], z3_samples[j])
#     end
# end

# # strong solution tests
# strong_analytic_solution = [Particles(analytic_solution_samples[i, :])
#                             for i in eachindex(ts)]
# strong_truncated_solution = [Particles(truncated_solution_samples[i, :])
#                              for i in eachindex(ts)]
# strong_predicted_solution_1 = [Particles(predicted_solution_samples_1[i, :])
#                                for i in eachindex(ts)]
# strong_predicted_solution_2 = [Particles(predicted_solution_samples_2[i, :])
#                                for i in eachindex(ts)]

# error_1 = sum(abs2, strong_analytic_solution .- strong_predicted_solution_1)
# error_2 = sum(abs2, strong_analytic_solution .- strong_predicted_solution_2)
# @test pmean(error_1) > pmean(error_2)

# @test pmean(sum(abs2.(strong_predicted_solution_1 .- strong_truncated_solution))) >
#       pmean(sum(abs2.(strong_predicted_solution_2 .- strong_truncated_solution)))

# # weak solution tests
# mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
# mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
# mean_predicted_solution_1 = mean(predicted_solution_samples_1, dims = 2)
# mean_predicted_solution_2 = mean(predicted_solution_samples_2, dims = 2)

# # testing over different Z_i sample sizes
# error_1 = sum(abs2, mean_analytic_solution .- pmean(u1))
# error_2 = sum(abs2, mean_analytic_solution .- pmean(u2))
# @test error_1 > error_2

# MSE_1 = mean(abs2.(mean_analytic_solution .- pmean(u1)))
# MSE_2 = mean(abs2.(mean_analytic_solution .- pmean(u2)))
# @test MSE_2 < MSE_1
# @test MSE_2 < 5e-2

# error_1 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_1)
# error_2 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_2)
# @test error_1 > error_2

# MSE_1 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_1))
# MSE_2 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2))
# @test MSE_2 < MSE_1
# @test MSE_2 < 5e-2

# @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) >
#       mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution))
# @test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 3e-1
# @test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 4e-2
# # end

# # plotting
# using Plots
# plotly()
# plot(ts, strong_analytic_solution, label = "analytic sol",
#     legend = :outerbottomright, title = "GBM SDE Strong Sol")
# plot!(ts, strong_truncated_solution, label = "truncated sol")
# plot!(ts, u1, label = "2000 test samples, 1 z-train subsample")
# plot!(ts, strong_predicted_solution_1, label = "3000 test samples, 1 z-train subsample")
# plot!(ts, u2, label = "2000 test samples, 10 z-train subsample")
# plot!(ts, strong_predicted_solution_2, label = "3000 test samples, 10 z-train subsamples")

# # plotting
# plot(ts, mean_analytic_solution, label = "analytic sol",
#     legend = :outerbottomright, title = "GBM SDE Weak Sol")
# plot!(ts, mean_truncated_solution, label = "truncated sol")
# plot!(ts, pmean(u1), label = "2000 test samples, 1 z-train subsample")
# plot!(ts, mean_predicted_solution_1, label = "3000 test samples, 1 z-train subsample")
# plot!(ts, pmean(u2), label = "2000 test samples, 10 z-train subsample")
# plot!(ts, mean_predicted_solution_2, label = "3000 test samples, 10 z-train subsamples")

# using Plots
# plotly()
# # plotting sols
# plot(ts, strong_analytic_solution, label = "analytic sol",
#     legend = :outerbottomright, title = "Additive noise Test SDE Strong Sol")
# plot!(ts, strong_truncated_solution, label = "truncated sol")
# plot!(ts, u1, label = "2000 test samples, 1 z-train subsample")
# plot!(ts, strong_predicted_solution_1, label = "3000 test samples, 1 z-train subsample")
# plot!(ts, u2, label = "2000 test samples, 10 z-train subsample")
# plot!(ts, strong_predicted_solution_2, label = "3000 test samples, 10 z-train subsamples")

# # plotting sols
# plot(ts, mean_analytic_solution, label = "analytic sol",
# legend = :outerbottomright, title = "Additive noise Test SDE Weak Sol")
# plot!(ts, mean_truncated_solution, label = "truncated sol")
# plot!(ts, u1, label = "2000 test samples, 1 z-train subsample")
# plot!(ts, mean_predicted_solution_1, label = "3000 test samples, 1 z-train subsample")
# plot!(ts, u2, label = "2000 test samples, 10 z-train subsample")
# plot!(ts, mean_predicted_solution_2, label = "3000 test samples, 10 z-train subsamples")
