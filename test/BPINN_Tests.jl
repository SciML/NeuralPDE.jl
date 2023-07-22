# Testing out Code (will put in bpinntests.jl file in tests directory)

# test code for Lux chains hasnt been added yet
using DifferentialEquations, MCMCChains, ForwardDiff, Distributions
using NeuralPDE, Flux, OptimizationOptimisers, AdvancedHMC
using StatProfilerHTML, Profile, Statistics
using BenchmarkTools, Plots, StatsPlots
plotly()
Profile.init()

# PROBLEM-1
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)
tspan = (0.0, 2.0)
u0 = 0.0
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# Numerical and Analytical Solutions
ta = range(tspan[1], 1.0, length = 300)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
sol1 = solve(prob, Tsit5())

# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]

# Call BPINN, create chain
chainfh = Flux.Chain(Dense(1, 5, tanh), Dense(5, 1))
fh_mcmc_chain, fhsamples, fhstats = ahmc_bayesian_pinn_ode(prob, chainfh, dataset,
                                                           draw_samples = 2000)

init, re = destructure(chainfh)

t = range(tspan[1], 8, length = 400)
time = vec(collect(Float64, t))
t = time
p = prob.p

# ALL SAMPLES PLOTS--------------------------------------
# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

# Plot each iv'th sample in samples
function realsol(out, t, iv)
    outpreds = out(t')
    physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* vec(outpreds)
    plot!(t, physsol3, label = "y(x,t) $iv th curve")
end

# Plot all NN parameter samples curve posibilities
for i in eachindex(fhsamples)
    out = re(fhsamples[i])
    realsol(out, t, i)
end

# added this as above plot takes time this lower line renders the above plot faster
plot!(title = "Problem1 y'(x,t),y(x,t) for ODE,B PINN", legend = :outerbottomright)

# PLOT MCMC Chain
plot(fh_mcmc_chain)

# # PLOTTING MEANS AND iTH PARAMETER CURVES
# # Plot problem and its solution
# plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
# physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
# physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
# plot!(t, physsol1, label = "y(x,t)")
# plot!(t, physsol2, label = "y'(x,t)")

# means = mean(matrix_samples, dims = 2)
# medians = median(matrix_samples, dims = 2)

# # plotting average of final nn outputs
# out = re.(fhsamples)
# yu = collect(out[i](t') for i in eachindex(out))
# yu = vcat(yu...)
# a = [mean(yu[:, i]) for i in eachindex(t)]
# plot!(t, prob.u0 .+ (t .- prob.tspan[1]) .* a, label = "curve averages")

# # plotting i'th sampled parameters NN output
# a = vec(re(vec(fhsamples[1000]))(t'))
# physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
# plot!(t, physsol3, label = "y(x,t) 1000th curve")

# # plotting curve when using mean of sampled parameters
# a = vec(re(vec(means))(t'))
# physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
# plot!(t, physsol3, label = "y(x,t) means curve")

# PROBLEM-2 LOTKA VOLTERRA EXAMPLE
function lotka_volterra(u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    dx = (α - β * y) * x # prey
    dy = (δ * x - γ) * y # predator

    return [dx, dy]
end

u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)
solution = solve(prob, Tsit5(); saveat = 0.1)
# Plot simulation.
plot(solution)
time = solution.t
u = hcat(solution.u...)
# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x = u[1, :] + 0.5 * randn(length(u[1, :]))
y = u[2, :] + 0.5 * randn(length(u[1, :]))
dataset = [x, y, time]
# scatter!(time, [x, y])

chainfh = Flux.Chain(Dense(1, 8, sigmoid_fast), Dense(8, 2))
# chainfh = Flux.Chain(Dense(1, 5, sigmoid), Dense(5, 1))

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainfh, dataset,
                                                              draw_samples = 2000,
                                                              l2std = [0.05, 0.05],
                                                              phystd = [0.05, 0.05],
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [
                                                                  Normal(1.5, 0.5),
                                                                  Normal(1.2, 0.5),
                                                                  Normal(3.3, 0.5),
                                                                  Normal(1.4, 0.5),
                                                              ])

init, re = destructure(chainfh)

a = re(fhsamples1[2000][1:34])(time')
physsol3 = prob.u0 .+ (time' .- prob.tspan[1]) .* a
plot!(time, physsol3[2, :], label = "lotka(0,10)2000")

p2 = fhsamples1[2000][35]
p3 = fhsamples1[2000][36]
p4 = fhsamples1[2000][37]
p5 = fhsamples1[2000][38]
# julia > p2 = fhsamples1[1000][35]
# 1.7432577621468217

# julia > p3 = fhsamples1[1000][36]
# 1.1105561135057131

# julia > p4 = fhsamples1[1000][37]
# 2.205520643590177

# julia > p5 = fhsamples1[1000][38]
# 0.7368208188492641


p2 = fhsamples1[1000][35]
p3 = fhsamples1[1000][36]
p4 = fhsamples1[1000][37]
p5 = fhsamples1[1000][38]
# PROBLEM-1 (WITH PARAMETER ESTIMATION)
linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
linear = (u, p, t) -> cos(p * t)
tspan = (0.0, 2.0)
u0 = 0.0
p = 2 * pi
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, p)

# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
# Numerical and Analytical Solutions
sol1 = solve(prob, Tsit5(); saveat = 0.01)
u = sol1.u
time = sol1.t
# plot(sol.t,sol.u)

# Numerical and Analytical Solutions
ta = range(tspan[1], tspan[2], length = 200)
u = [linear_analytic(u0, p, ti) for ti in ta]

# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
# plot!(time, x̂)

chainfh1 = Flux.Chain(Dense(1, 5, tanh), Dense(5, 1))
chainfh2 = Flux.Chain(Dense(1, 10, tanh), Dense(10, 1))

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainfh1, dataset,
                                                              draw_samples = 2000,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 2)],
                                                              Metric = DiagEuclideanMetric)

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainfh2, dataset,
                                                              draw_samples = 2000,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(7, 1.5)],
                                                              Metric = DiagEuclideanMetric)

init1, re1 = destructure(chainfh1)
init2, re2 = destructure(chainfh2)

#   PLOT testing points 0-8
t = vec(collect(Float64, range(tspan[1], 8, length = 800)))

# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN with param", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

# # Create mcmc chain
# samples = fhsamples1
# matrix_samples = hcat(samples...)
# # fh_mcmc_chain = Chains(matrix_samples')  # Create a chain from the reshaped samples
# fh_mcmc_chain
# means = mean(matrix_samples, dims = 2)
# # plotting average of final nn outputs
# out = re.(fhsamples1)
# yu = collect(out[i](t') for i in eachindex(out))
# yu = vcat(yu...)
# a = [mean(yu[:, i]) for i in eachindex(t)]
# plot!(t, prob.u0 .+ (t .- prob.tspan[1]) .* a, label = "curve averages")

# plotting i'th sampled parameters NN output
a = vec(re1(fhsamples1[2000][1:16])(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3,
      label = "full(tspan[2],3per,8)(200,0.02)DiagoverlapLogparam(9,4)(1,5,1)")

a = vec(re2(fhsamples2[2000][1:31])(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3,
      label = "full(tspan[2],3per,8)(200,0.02)DiagoverlapLogparam(7,1.5)(1,10,1)")

# newrun till 2
p2 = fhsamples1[2000][17]
p1 = fhsamples2[2000][32]

fh_mcmc_chain1
summarize(fh_mcmc_chain2[[:param_32]])
fh_mcmc_chain2
summarize(fh_mcmc_chain2[[:param_32]])

physsol1 = [linear_analytic(prob.u0, p2, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p2, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t) p2")
plot!(t, physsol2, label = "y'(x,t) p2")

# PROBLEM-3
linear = (u, p, t) -> -u / 5 + exp(-t + p[1] / 5) * cos(t) / p[2]
tspan = (0.0, 10.0)
u0 = 0.0
p = [2.0, 5.0]
prob = ODEProblem(linear, u0, tspan, p)

sol1 = solve(prob, Tsit5(); saveat = 0.1)
u = sol1.u[1:50]
time = sol1.t[1:50]
# plot(sol1.t, sol1.u)

x̂ = collect(Float64, Array(u) + 0.005 * randn(size(u)))
dataset = [x̂, time]
# plot!(time, x̂)
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# new destructure does not affect
# Hamiltonian energy must be lowest(more paramters the better is it to map onto them)
# full better than L2 and phy individual(test)
# in mergephys more points after training points is better from 20->40
# does consecutive runs bceome better? why?(plot 172)(same chain maybe)
# does density of points in timespan matter dataset vs internal timespan?(plot 172)(100+0.01)
# when training from 0-1 and phys from 1-5 with 1/150 simple nn slow,but bigger nn faster decrease in Hmailtonian
# bigger time interval more curves to adapt to only more parameters adapt to that, better NN architecture
# higher order logproblems solve better
# repl up up are same instances? but reexecute calls are new?

# PROBLEM-3
# linear = (u, p, t) -> -u / 5 + exp(-t / 5) * cos(t)
# linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
# tspan = (0.0f0, 10.0f0)
# u0 = 0.0f0
# prob = ODEProblem(ODEFunction(linear, analytic=linear_analytic), u0, tspan)