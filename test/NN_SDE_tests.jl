# @testitem "Test-1" tags=[:nnsde] begin
using OrdinaryDiffEq, Random, Lux, Optimisers
using DifferentialEquations
using OptimizationOptimJL: BFGS
Random.seed!(100)

α = 1.2
β = 1.1
u₀ = 1.0
f(u, p, t) = α * u
g(u, p, t) = β * u
tspan = (0.0, 1.0)
prob = SDEProblem(f, g, u₀, tspan)
dim = 1 + 3
luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))
dt = 1 / 50.0f0
abstol = 1e-10
autodiff = false
kwargs = (; verbose = true, dt = dt, abstol, maxiters = 2000)
# opt = Adam(0.1)

opt = BFGS()
#  Adam(0.05)
numensemble = 300

using NeuralPDE, Distributions
sol_1 = solve(
    prob, NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true);
    kwargs...)
sol_2 = solve(
    prob, NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 5, batch = true);
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

num_samples = 200
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

using Plots, Printf, Test
plotly()
p1 = plot(ts, mean_predicted_solution_1, title = @sprintf("PINN Predicted Solution"))
p2 = plot(ts, mean_analytic_solution, title = @sprintf("Analytic Solution"))
p3 = plot(ts, mean_truncated_solution, title = @sprintf("Truncated Solution"))
p4 = plot(ts, mean_predicted_solution_2, title = @sprintf("PINN Predicted Solution"))

plot(p1, p3, p2, p4, legend = :outerbottomright)

@test sum(abs2, mean_analytic_solution .- mean_truncated_solution) < 0.25

error_1 = sum(abs2, mean_analytic_solution .- u1)
error_2 = sum(abs2, mean_analytic_solution .- u2)
@test error_1 > error_2

error_1 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_1)
error_2 = sum(abs2, mean_analytic_solution .- mean_predicted_solution_2)
@test error_1 > error_2

MSE_1 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_1))
MSE_2 = mean(abs2.(mean_analytic_solution .- mean_predicted_solution_2))
@test MSE_2 < MSE_1
@test MSE_1 < 1e-2

@test mean(abs2.(mean_predicted_solution_1 .- mean_truncated_solution)) < 1e-2
@test mean(abs2.(mean_predicted_solution_2 .- mean_truncated_solution)) < 1e-2

#     @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
#         autodiff in [false, true]

#         if autodiff
#             @test_throws ArgumentError solve(
#                 prob, NNSDE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0)
#             continue
#         end

#         @testset for (dt, abstol) in [(1 / 20.0f0, 1e-10), (nothing, 1e-6)]
#             kwargs = (; verbose = false, dt, abstol, maxiters = 200)
#             sol = solve(prob, NNSDE(luxchain, opt; autodiff); kwargs...)
#         end
#     end
# end

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# @testitem "Test-1" tags=[:nnsde] begin
using OrdinaryDiffEq, Random, Lux
using NeuralPDE, DifferentialEquations
using OptimizationOptimJL: BFGS
using Optimisers
using Plots, Distributions, Printf
plotly()
Random.seed!(100)

α = 1.2
β = 1.1
u₀ = 1.0

f(u, p, t) = α * u
g(u, p, t) = β * u
tspan = (0.0, 1.0)
prob = SDEProblem(f, g, u₀, tspan)
dim = 1 + 3
luxchain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))
# θinit, st = Lux.setup(Random.default_rng(), luxchain)

# luxchain1 = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 2))
# θinit, st = Lux.setup(Random.default_rng(), luxchain1)
# a = luxchain([1, 2, 3, 4], θinit, st)[1]
# b = luxchain([1, 1, 1, 1], θinit, st)[1]

dt = 1 / 50.0f0
abstol = 1e-10
autodiff = false
kwargs = (; verbose = true, dt = dt, abstol, maxiters = 1500)
# Stochastic optimisrs overs Deterministic in case of randn gen within inner loss(stochastic objective), but this fails as optimization is too tough(infinite objectives possible)
# how is PINN prev attemp solver, smooth?

# something related to accuracy and dt, numensemble is happening
# if trained dt and tested dt is same i get good fits
# if the dts are different i get bad fits, maybe exlusive to grid training - test therefore
opt = BFGS()
numensemble = 200
# if numenselbe is high, then error can be alot in comparison to means of truncated and analytic
sol = solve(
    prob, NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 2, batch = true);
    kwargs...)

sol_2 = solve(
    prob, NNSDE(
        luxchain, opt; autodiff, numensemble = numensemble, sub_batch = 1, batch = true);
    kwargs...)

# collect(tspan[1]:dt:tspan[2])
ts = sol.timepoints
u = sol.mean_fit
plot!(ts, u, legend = :bottomright)
plot(sol_2.timepoints, sol_2.mean_fit, legend = :bottomright)

analytic_sol(u0, p, t, W) = u0 * exp((α - β^2 / 2) * t + β * W)
u0 = 1.0

function W_kkl(t, z1, z2, z3)
    √2 * (z1 * sin((1 - 1 / 2) * π * t) / ((1 - 1 / 2) * π) +
     z2 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π) +
     z3 * sin((2 - 1 / 2) * π * t) / ((2 - 1 / 2) * π))
end
truncated_sol(u0, t, z1, z2, z3) = u0 * exp((α - β^2 / 2) * t + β * W_kkl(t, z1, z2, z3))

num_samples = 500
num_time_steps = dt
z1_samples = rand(Normal(0, 1), num_samples)
z2_samples = rand(Normal(0, 1), num_samples)
z3_samples = rand(Normal(0, 1), num_samples)
ts1 = sol.timepoints

num_time_steps = size(ts1)[1]
W_samples = Array{Float64}(undef, num_time_steps, num_samples)
for i in 1:num_samples
    W = WienerProcess(0.0, 0.0)
    probtemp = NoiseProblem(W, (0, 1))
    sol2 = solve(probtemp; dt = dt)
    W_samples[:, i] = sol2.u
end

# for each rand set we must check all timepoints, and then for all timepoints.
temp_rands = hcat(z1_samples, z2_samples, z3_samples)'
inputs_test = [hcat([vcat(ts1[j], temp_rands[:, i]) for j in eachindex(ts1)]...)
               for i in 1:num_samples]

analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
predicted_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)

# here inputs do not match inputs over which pinn was optimized, we basically do testing over a large set than training set
# how much larger? thats decided by the numensemble/num_samples
for j in 1:num_samples
    for i in 1:num_time_steps
        # for each sample, pass each timepoints and get output
        analytic_solution_samples[i, j] = analytic_sol(u0, 0, ts1[i], W_samples[i, j])
        predicted_solution_samples[i, j] = prob.u0 .+
                                           (ts1[i] - prob.tspan[1]) .*
                                           sol.solution.interp.phi(
            inputs_test[j][:, i], sol.solution.interp.θ)
        truncated_solution_samples[i, j] = truncated_sol(
            u0, ts1[i], z1_samples[j], z2_samples[j], z3_samples[j])
    end
end

mean_analytic_solution = mean(analytic_solution_samples, dims = 2)
mean_truncated_solution = mean(truncated_solution_samples, dims = 2)
mean_predicted_solution = mean(predicted_solution_samples, dims = 2)
sum(abs2, mean_analytic_solution .- mean_predicted_solution)
sum(abs2, mean_analytic_solution .- mean_truncated_solution)
sum(abs2, mean_predicted_solution .- mean_truncated_solution)
sum(abs2, mean_analytic_solution .- sol.mean_fit)

# Diffeqarray is what blud it causies PINNsolution to be smooth, curvatiure in truncated solution is coming from num_samples

p1 = plot(ts1, mean_predicted_solution, title = @sprintf("PINN Predicted Solution"))
p2 = plot(ts1, mean_analytic_solution, title = @sprintf("Analytic Solution"))
p3 = plot(ts1, mean_truncated_solution, title = @sprintf("Truncated Solution"))
plot(p1, p3, p2, legend = :bottomright)

analytic_solution_samples1 = Array{Float64}(undef, num_time_steps, num_samples)
truncated_solution_samples1 = Array{Float64}(undef, num_time_steps, num_samples)
predicted_solution_samples1 = Array{Float64}(undef, num_time_steps, num_samples)

# here inputs do not match inputs over which pinn was optimized, we basically do testing over a large set than training set
# how much larger? thats decided by the numensemble/num_samples
for j in 1:num_samples
    for i in 1:num_time_steps
        # for each sample, pass each timepoints and get output
        analytic_solution_samples1[i, j] = analytic_sol(u0, 0, ts1[i], W_samples[i, j])
        predicted_solution_samples1[i, j] = prob.u0 .+
                                            (ts1[i] - prob.tspan[1]) .*
                                            sol_2.solution.interp.phi(
            inputs_test[j][:, i], sol_2.solution.interp.θ)
        truncated_solution_samples1[i, j] = truncated_sol(
            u0, ts1[i], z1_samples[j], z2_samples[j], z3_samples[j])
    end
end

mean_analytic_solution1 = mean(analytic_solution_samples1, dims = 2)
mean_truncated_solution1 = mean(truncated_solution_samples1, dims = 2)
mean_predicted_solution1 = mean(predicted_solution_samples1, dims = 2)
sum(abs2, mean_analytic_solution1 .- mean_predicted_solution1)
sum(abs2, mean_analytic_solution1 .- mean_truncated_solution1)
sum(abs2, mean_predicted_solution1 .- mean_truncated_solution1)
sum(abs2, mean_analytic_solution1 .- sol_2.mean_fit)

# Diffeqarray is what blud it causies PINNsolution to be smooth, curvatiure in truncated solution is coming from num_samples

p1 = plot(ts1, mean_predicted_solution1, title = @sprintf("PINN Predicted Solution"))
p2 = plot(ts1, mean_analytic_solution1, title = @sprintf("Analytic Solution"))
p3 = plot(ts1, mean_truncated_solution1, title = @sprintf("Truncated Solution"))
plot(p1, p3, p2, legend = :bottomright)

# @testset "$(nameof(typeof(opt))) -- $(autodiff)" for opt in [BFGS(), Adam(0.1)],
#     autodiff in [false, true]

#     if autodiff
#         @test_throws ArgumentError solve(
#             prob, NNSDE(luxchain, opt; autodiff); maxiters = 200, dt = 1 / 20.0f0)
#         continue
#     end

#     @testset for (dt, abstol) in [(1 / 20.0f0, 1e-10), (nothing, 1e-6)]
#         kwargs = (; verbose = false, dt, abstol, maxiters = 200)
#         sol = solve(prob, NNSDE(luxchain, opt; autodiff); kwargs...)
#     end
# end
# end