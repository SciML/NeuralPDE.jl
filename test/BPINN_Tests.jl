# Testing out Code (will put in bpinntests.jl file in tests directory)

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
tspan = (0.0, 4.0)
u0 = 0.0
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# Numerical and Analytical Solutions
ta = range(tspan[1], 2.0, length = 300)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
sol1 = solve(prob, Tsit5())

# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x̂ = collect(Float64, Array(u) + 0.05 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = (x̂, time)

# Call BPINN, create chain
chainfh = Flux.Chain(Dense(1, 5, tanh), Dense(5, 1))
fh_mcmc_chain, fhsamples, fhstats = ahmc_bayesian_pinn_ode(prob, chainfh, dataset,
                                                           draw_samples = 1000)
init, re = destructure(chainfh)

t = range(tspan[1], tspan[2], length = 600)
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

# -------------------------------------------------------------------------------------------------------------------------
# (Above is the testing code as commented,use it in another env in case package issues arise)

# -------------------------------------------------------------------------------------------------------------------------

# NOTE:bayesian_pinn_ode for 500 points and 100 samples took >30mins and inaccurate results
# also default bayesian_pinn_ode call takes 3hrs ish

# for AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}
# ------------ Leapfrop ----------------
# STAdaptor(100,101)
# Sampling 100%|███████████████████████████████| Time: 0:22:36
#   iterations:                                   1000
#   ratio_divergent_transitions:                  0.05
#   ratio_divergent_transitions_during_adaption:  0.02
#   n_steps:                                      585
#   is_accept:                                    true
#   acceptance_rate:                              0.5811073129763328
#   log_density:                                  473.1202767415749
#   hamiltonian_energy:                           -463.7879654281731
#   hamiltonian_energy_error:                     0.7163532239870847
#   max_hamiltonian_energy_error:                 1432.1012241888504
#   tree_depth:                                   9
#   numerical_error:                              true
#   step_size:                                    0.0005088204439566235
#   nom_step_size:                                0.0005088204439566235
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...])
# ┌ Info: Finished 1000 sampling steps for 1 chains in 1356.8107155 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.000509), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.008942692170287743
# └   average_acceptance_rate = 0.6585250047374884

# VS

# NAdaptor(102,103)
# Sampling 100%|███████████████████████████████| Time: 0:20:19
#   iterations:                                   1000
#   ratio_divergent_transitions:                  0.42
#   ratio_divergent_transitions_during_adaption:  0.01
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.5459088063866748
#   log_density:                                  -299.76842331002837
#   hamiltonian_energy:                           309.113274342267
#   hamiltonian_energy_error:                     0.9525576789356478
#   max_hamiltonian_energy_error:                 9.281206296630273
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.0006998034273471476
#   nom_step_size:                                0.0006998034273471476
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.9482070590934292, 0.1734 ...])
# ┌ Info: Finished 1000 sampling steps for 1 chains in 1220.0076976 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.9482070590934292, 0.1734 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.0007), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.2770222858178922
# └   average_acceptance_rate = 0.4834599379989383

# Reduced training points-100 STAdaptor(104,106)
# Sampling 100%|███████████████████████████████| Time: 0:22:10
#   iterations:                                   1000
#   ratio_divergent_transitions:                  0.01
#   ratio_divergent_transitions_during_adaption:  0.01
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.9986235222022787
#   log_density:                                  -508.5775689167445
#   hamiltonian_energy:                           514.1073082731626
#   hamiltonian_energy_error:                     -0.3654598686513282
#   max_hamiltonian_energy_error:                 -0.6800779152758878
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.0006645345772298939
#   nom_step_size:                                0.0006645345772298939
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...])
# ┌ Info: Finished 1000 sampling steps for 1 chains in 1330.3505584 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.000665), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.24425221781740108
# └   average_acceptance_rate = 0.8558170208362209
# (Float32[-0.49601412, 0.0133395195, -0.32724154, 0.8161284, -0.21146107, 0.0, 0.0, 0.0, 0.0, 0.0, -0.37794662, -0.5061618, 0.73520625, 0.97469485, -0.66954374, 0.0], 
# # Restructure(Chain, ..., 16))

# Reduced training points-100 naDAPTERAdaptor(100,101)

# after adding mutable structs for 300 train points (110-) for 100 train points()

# ----------- Jiteered Leapffrog ---------------

# ------------ Tempered Leapfrog ----------------------------
