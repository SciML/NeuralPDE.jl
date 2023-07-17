# Testing out Code (will put in bpinntests.jl file in tests directory)

# test code for Lux chains hasnt been added yet

# Define ODE problem,conditions and Model,solve using NNODE.
using DifferentialEquations, MCMCChains, ForwardDiff
using NeuralPDE, Flux, OptimizationOptimisers
using StatProfilerHTML, Profile, Statistics
using BenchmarkTools, Plots, StatsPlots
plotly()
Profile.init()

# # # prob1
# linear = (u, p, t) -> -u / 5 + exp(-t / 5) * cos(t)
# linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
# tspan = (0.0f0, 10.0f0)
# u0 = 0.0f0
# prob = ODEProblem(ODEFunction(linear, analytic=linear_analytic), u0, tspan)

# prob2
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
dataset = (x̂, time)

# Call BPINN, create chain
chainfh = Flux.Chain(Dense(1, 5, tanh), Dense(5, 1))
fh_mcmc_chain, fhsamples, fhstats = ahmc_bayesian_pinn_ode(prob, chainfh, dataset,
                                                           draw_samples = 700,
                                                           autodiff)
fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainfh, dataset,
                                                              draw_samples = 700,
                                                              autodiff = true)

init, re = destructure(chainfh)

t = range(tspan[1], 4, length = 600)
time = vec(collect(Float64, t))
t = time
p = prob.p

# PLOTTING MEANS AND iTH PARAMETER CURVES
# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

means = mean(matrix_samples, dims = 2)
medians = median(matrix_samples, dims = 2)

# plotting average of final nn outputs
out = re.(fhsamples)
yu = collect(out[i](t') for i in eachindex(out))
yu = vcat(yu...)
a = [mean(yu[:, i]) for i in eachindex(t)]
plot!(t, prob.u0 .+ (t .- prob.tspan[1]) .* a, label = "curve averages")

# plotting i'th sampled parameters NN output
a = vec(re(vec(fhsamples[1000]))(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3, label = "y(x,t) 1000th curve")

# plotting curve when using mean of sampled parameters
a = vec(re(vec(means))(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3, label = "y(x,t) means curve")

# ALL SAMPLES PLOTS--------------------------------------
# Plot problem and its solution
# using numerical derivatives
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

# using autodiff
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

for i in eachindex(fhsamples1)
    out = re(fhsamples1[i])
    realsol(out, t, i)
end

# added this as above plot takes time this lower line renders the above plot faster
plot!(title = "Problem1 y'(x,t),y(x,t) for ODE,B PINN", legend = :outerbottomright)

# PLOT MCMC Chain
plot(fh_mcmc_chain1)