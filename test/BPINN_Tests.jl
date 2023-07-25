# Testing Code
using DifferentialEquations, MCMCChains, ForwardDiff, Distributions
using NeuralPDE, Flux, OptimizationOptimisers, AdvancedHMC, Lux
using StatProfilerHTML, Profile, Statistics
using BenchmarkTools, Plots, StatsPlots
plotly()
Profile.init()

# PROBLEM-1 (WITHOUT PARAMETER ESTIMATION)
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
chainflux = Flux.Chain(Flux.Dense(1, 5, tanh), Flux.Dense(5, 1))
chainlux = Lux.Chain(Lux.Dense(1, 5, tanh), Lux.Dense(5, 1))

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux, dataset,
                                                              draw_samples = 2000)
fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux, dataset,
                                                              draw_samples = 2000)

init1, re1 = destructure(chainflux)
init2, re2 = destructure(chainlux)
# TESTING TIMEPOINTS TO PLOT ON
t = vec(collect(Float64, range(tspan[1], 8, length = 400)))
p = prob.p

# ALL SAMPLES PLOTS--------------------------------------
# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

# Plot iv'th sample in samples
function realsol(out, t, iv)
    outpreds = out(t')
    physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* vec(outpreds)
    plot!(t, physsol3, label = "y(x,t) $iv th curve")
end

# Plot all NN parameter samples curve posibilities
for i in eachindex(fhsamples1)
    out = re1(fhsamples1[i])
    realsol(out, t, i)
end

# added this as above plot takes time this lower line renders the above plot faster
plot!(title = "Problem1 y'(x,t),y(x,t) for ODE,B PINN", legend = :outerbottomright)

# PLOT MCMC Chain
plot(fh_mcmc_chain1)

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

# PROBLEM-1 (WITH PARAMETER ESTIMATION)
linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
linear = (u, p, t) -> cos(p * t)
tspan = (0.0, 2.0)
u0 = 0.0
p = 2 * pi
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, p)

# Numerical and Analytical Solutions
sol1 = solve(prob, Tsit5(); saveat = 0.01)
u = sol1.u
time = sol1.t
# plot(sol1.t, sol1.u)

# BPINN AND TRAINING DATASET CREATION
ta = range(tspan[1], tspan[2], length = 200)
u = [linear_analytic(u0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
plot!(time, x̂)

# comparing how diff NNs capture non-linearity
chainflux1 = Flux.Chain(Flux.Dense(1, 5, tanh), Flux.Dense(5, 1))
chainlux1 = Lux.Chain(Lux.Dense(1, 5, tanh), Lux.Dense(5, 1))

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux1, dataset,
                                                              draw_samples = 2000,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 2)],
                                                              Metric = DiagEuclideanMetric)

fh_mcmc_chain2, fhsamples2, fhstats2 = ahmc_bayesian_pinn_ode(prob, chainlux1, dataset,
                                                              draw_samples = 2000,
                                                              physdt = 1 / 50.0f0,
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [LogNormal(9, 2)],
                                                              Metric = DiagEuclideanMetric)

init1, re1 = destructure(chainflux1)
init2, re2 = destructure(chainlux1)

#   PLOT testing points 0-8
t = vec(collect(Float64, range(tspan[1], 8, length = 800)))

# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN with param", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

# plotting LAST sampled parameters NN1 output
a = vec(re1(fhsamples1[2000][1:16])(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3,
      label = "full(tspan[2],3per,8)(200,0.02)DiagoverlapLogparam(9,4)(1,5,1)")

# plotting LAST sampled parameters NN2 output
a = vec(re2(fhsamples2[2000][1:31])(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3,
      label = "full(tspan[2],3per,8)(200,0.02)DiagoverlapLogparam(7,1.5)(1,10,1)")

# estimated ODE parameters NN1 AND NN2
p1 = fhsamples1[2000][17]
p2 = fhsamples2[2000][32]

# CHAINS STATS
fh_mcmc_chain1
summarize(fh_mcmc_chain2[[:param_32]])
fh_mcmc_chain2
summarize(fh_mcmc_chain2[[:param_32]])

# ODE CURVE ON THE ESTIMATED ODE PARAMETERS (NN1 AND NN2)
physsol1 = [linear_analytic(prob.u0, p1, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p1, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t) p1")
plot!(t, physsol2, label = "y'(x,t) p1")

physsol1 = [linear_analytic(prob.u0, p2, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p2, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t) p2")
plot!(t, physsol2, label = "y'(x,t) p2")

# PROBLEM-2 LOTKA VOLTERRA EXAMPLE (WITH PARAMETER ESTIMATION)
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
dataset = [x[1:60], y[1:60], time[1:60]]
scatter!(time, [x, y])

# NN has 2 outputs as u -> [dx,dy]
chainflux1 = Lux.Chain(Lux.Dense(1, 8, Lux.tanh), Lux.Dense(8, 8, Lux.tanh),
                       Lux.Dense(8, 2))
chainlux1 = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 8, tanh), Flux.Dense(8, 2))

fh_mcmc_chainflux1, fhsamplesflux1, fhstatsflux1 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                                          dataset,
                                                                          draw_samples = 1000,
                                                                          l2std = [
                                                                              0.05,
                                                                              0.05,
                                                                          ],
                                                                          phystd = [
                                                                              0.05,
                                                                              0.05,
                                                                          ],
                                                                          priorsNNw = (0.0,
                                                                                       3.0))

fh_mcmc_chainflux2, fhsamplesflux2, fhstatsflux2 = ahmc_bayesian_pinn_ode(prob, chainflux1,
                                                                          dataset,
                                                                          draw_samples = 1000,
                                                                          l2std = [
                                                                              0.05,
                                                                              0.05,
                                                                          ],
                                                                          phystd = [
                                                                              0.05,
                                                                              0.05,
                                                                          ],
                                                                          priorsNNw = (0.0,
                                                                                       3.0),
                                                                          param = [
                                                                              Normal(1.5,
                                                                                     0.5),
                                                                              Normal(1.2,
                                                                                     0.5),
                                                                              Normal(3.3,
                                                                                     0.5),
                                                                              Normal(1.4,
                                                                                     0.5),
                                                                          ])
fh_mcmc_chainlux1, fhsampleslux1, fhstatslux1 = ahmc_bayesian_pinn_ode(prob, chainlux1,
                                                                       dataset,
                                                                       draw_samples = 1000,
                                                                       l2std = [0.05, 0.05],
                                                                       phystd = [
                                                                           0.05,
                                                                           0.05,
                                                                       ],
                                                                       priorsNNw = (0.0,
                                                                                    3.0))

fh_mcmc_chainlux2, fhsampleslux2, fhstatslux2 = ahmc_bayesian_pinn_ode(prob, chainlux1,
                                                                       dataset,
                                                                       draw_samples = 1000,
                                                                       l2std = [0.05, 0.05],
                                                                       phystd = [
                                                                           0.05,
                                                                           0.05,
                                                                       ],
                                                                       priorsNNw = (0.0,
                                                                                    3.0),
                                                                       param = [
                                                                           Normal(1.5, 0.5),
                                                                           Normal(1.2, 0.5),
                                                                           Normal(3.3, 0.5),
                                                                           Normal(1.4, 0.5),
                                                                       ])

init1, re1 = destructure(chainflux1)
init2, re2 = destructure(chainlux1)

# PLOT NN SOLUTIONS
a = re(fhsamples1[2000][1:106])(time')
physsol3 = prob.u0 .+ (time' .- prob.tspan[1]) .* a
plot!(time, physsol3[1, :], label = "1882tanhlotka(0,10)1ordersolveode2000")
plot!(time, physsol3[2, :], label = "1882tanhlotka(0,10)1ordersolveode2000")

# ESTIMATED ODE PARAMETER VALUES
p2 = fhsamplesflux2[2000][107]
p3 = fhsamplesflux2[2000][108]
p4 = fhsamplesflux2[2000][109]
p5 = fhsamplesflux2[2000][110]

# SECOND ORDER
# Sampling 100%|███████████████████████████████| Time: 0:17:30
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.77
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    false
#   acceptance_rate:                              0.0
#   log_density:                                  -67011.71340625182
#   hamiltonian_energy:                           67060.85603778006
#   hamiltonian_energy_error:                     0.0
#   numerical_error:                              true
#   step_size:                                    0.0017987463400432056
#   nom_step_size:                                0.0017987463400432056
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00012814204741063516, 0. ...])
# ┌ Info: Finished 2000 sampling steps for 1 chains in 1050.3880364 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00012814204741063516, 0. ...]), kinetic=GaussianKinetic())
# │   κ = HMCKernel{AdvancedHMC.FullMomentumRefreshment, Trajectory{EndPointTS, Leapfrog{Float64}, FixedNSteps}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{EndPointTS}(integrator=Leapfrog(ϵ=0.0018), tc=FixedNSteps(50)))
# │   EBFMI_est = 0.13837956735052787
# └   average_acceptance_rate = 0.31602309542818136

# julia > p2 = fhsamples1[2000][107]
# 1.6683238154450155

# julia > p3 = fhsamples1[2000][108]
# 1.0601120033414857

# julia > p4 = fhsamples1[2000][109]
# 2.4417596607369187

# julia > p5 = fhsamples1[2000][110]
# 0.7926392826347816

# Sampling 100%|███████████████████████████████| Time: 0:14:32
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    true
#   acceptance_rate:                              1.0
#   log_density:                                  -90906.18749660898
#   hamiltonian_energy:                           90975.24099621628
#   hamiltonian_energy_error:                     -0.20816170917532872
#   numerical_error:                              true
#   step_size:                                    0.0004538068757475764
#   nom_step_size:                                0.0004538068757475764
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00027672822844863925, 9. ...])
# ┌ Info: Finished 2000 sampling steps for 1 chains in 872.264636 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00027672822844863925, 9. ...]), kinetic=GaussianKinetic())
# │   κ = HMCKernel{AdvancedHMC.FullMomentumRefreshment, Trajectory{EndPointTS, Leapfrog{Float64}, FixedNSteps}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{EndPointTS}(integrator=Leapfrog(ϵ=0.000454), tc=FixedNSteps(50)))
# │   EBFMI_est = 0.14181129580900637
# └   average_acceptance_rate = 0.8029925128424785
# julia> p2 = fhsamples1[2000][107]
# 1.5392561124682074

# julia> p3 = fhsamples1[2000][108]
# 1.0342500889174497

# julia> p4 = fhsamples1[2000][109]
# 2.834987499816353

# julia> p5 = fhsamples1[2000][110]
# 0.9157470369793699

# FIRST ORDER
# Sampling 100%|███████████████████████████████| Time: 0:15:30
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.08
#   n_steps:                                      50
#   is_accept:                                    true
#   acceptance_rate:                              1.0
#   log_density:                                  -85708.5171167773
#   hamiltonian_energy:                           85771.90811152117
#   hamiltonian_energy_error:                     -0.5584391979355132
#   numerical_error:                              true
#   step_size:                                    0.0006952321398964403
#   nom_step_size:                                0.0006952321398964403
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00033528245219679575, 0. ...])
# ┌ Info: Finished 2000 sampling steps for 1 chains in 930.8274946 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00033528245219679575, 0. ...]), kinetic=GaussianKinetic())
# │   κ = HMCKernel{AdvancedHMC.FullMomentumRefreshment, Trajectory{EndPointTS, Leapfrog{Float64}, FixedNSteps}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{EndPointTS}(integrator=Leapfrog(ϵ=0.000695), tc=FixedNSteps(50)))
# │   EBFMI_est = 0.1377914790270954
# └   average_acceptance_rate = 0.7846144174763487
# julia > p2 = fhsamples1[2000][107]
# 1.5890976361646683

# julia > p3 = fhsamples1[2000][108]
# 1.0386135772823453

# julia > p4 = fhsamples1[2000][109]
# 2.6589517117646784

# julia > p5 = fhsamples1[2000][110]
# 0.8546692120183275

# 2nd run
# Sampling 100%|███████████████████████████████| Time: 0:14:35
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    true
#   acceptance_rate:                              0.5546957473024191
#   log_density:                                  -87895.81518995497
#   hamiltonian_energy:                           88048.44666048267
#   hamiltonian_energy_error:                     0.5893355186126428
#   numerical_error:                              true
#   step_size:                                    0.0007612502250460531
#   nom_step_size:                                0.0007612502250460531
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00018991466469841698, 0. ...])
# ┌ Info: Finished 2000 sampling steps for 1 chains in 875.9443404 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00018991466469841698, 0. ...]), kinetic=GaussianKinetic())
# │   κ = HMCKernel{AdvancedHMC.FullMomentumRefreshment, Trajectory{EndPointTS, Leapfrog{Float64}, FixedNSteps}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{EndPointTS}(integrator=Leapfrog(ϵ=0.000761), tc=FixedNSteps(50)))
# │   EBFMI_est = 0.13498155037293352
# └   average_acceptance_rate = 0.4839693508702932

# julia > p2 = fhsamples1[2000][107]
# 1.5651839495840574

# julia > p3 = fhsamples1[2000][108]
# 1.0346676685896596

# julia > p4 = fhsamples1[2000][109]
# 2.8396303437107013

# julia > p5 = fhsamples1[2000][110]
# 0.9085294697363377

# EXTRAPOLATOIN
a = re(fhsamples1[2000][1:106])(time')
physsol3 = prob.u0 .+ (time' .- prob.tspan[1]) .* a
plot!(time, physsol3[1, :], label = "1882tanhlotka(0,6,10)1order2000-2")
plot!(time, physsol3[2, :], label = "1882tanhlotka(0,6,10)1order2000-2")

# ESTIMATED ODE PARAMETER VALUES
p2 = fhsampleslux2[2000][107]
p3 = fhsampleslux2[2000][108]
p4 = fhsampleslux2[2000][109]
p5 = fhsampleslux2[2000][110]
# ESTIMATED ODE PARAMETER VALUES(PLOT 86 IS JUST TRAININED OVER A SMALLER TIMESPAN)
# julia > p2 = fhsamples1[2000][107]
# 1.4787077628364855

# julia > p3 = fhsamples1[2000][108]
# 1.021307053195241

# julia > p4 = fhsamples1[2000][109]
# 2.950374273954437

# julia > p5 = fhsamples1[2000][110]
# 1.0063845222623147

# real ESTIMATED ODE PARAMETER VALUES(trained till 6)
# Sampling 100%|███████████████████████████████| Time: 0:21:33
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    false
#   acceptance_rate:                              0.0
#   log_density:                                  -54545.15848124605
#   hamiltonian_energy:                           54595.72866640587
#   hamiltonian_energy_error:                     0.0
#   numerical_error:                              true
#   step_size:                                    0.0038562400207858543
#   nom_step_size:                                0.0038562400207858543
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([9.289170598084067e-5, 0.00 ...])
# ┌ Info: Finished 2000 sampling steps for 1 chains in 1295.0873309 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([9.289170598084067e-5, 0.00 ...]), kinetic=GaussianKinetic())
# │   κ = HMCKernel{AdvancedHMC.FullMomentumRefreshment, Trajectory{EndPointTS, Leapfrog{Float64}, FixedNSteps}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{EndPointTS}(integrator=Leapfrog(ϵ=0.00386), tc=FixedNSteps(50)))
# │   EBFMI_est = 0.8863679766107189
# └   average_acceptance_rate = 0.13640265742673682
# julia > p2 = fhsamples1[2000][107]
# 1.349546129803678

# julia > p3 = fhsamples1[2000][108]
# 0.789549189503873

# julia > p4 = fhsamples1[2000][109]
# 0.07158305700422583

# julia > p5 = fhsamples1[2000][110]
# 0.018736070548473125

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainlux1, dataset,
                                                              draw_samples = 2000,
                                                              l2std = [0.05, 0.05],
                                                              phystd = [0.05, 0.05],
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [
                                                                  Normal(1.5, 0.5),
                                                                  Normal(1.2, 0.5),
                                                                  Normal(3.3, 0.5),
                                                                  Normal(1.4, 0.5),
                                                              ], autodiff = true)

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux1, dataset,
                                                              draw_samples = 2000,
                                                              l2std = [0.05, 0.05],
                                                              phystd = [0.05, 0.05],
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [
                                                                  Normal(1.5, 0.5),
                                                                  Normal(1.2, 0.5),
                                                                  Normal(3.3, 0.5),
                                                                  Normal(1.4, 0.5),
                                                              ], autodiff = true)

#  Sampling 100%|███████████████████████████████| Time: 0:48:18
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    true
#   acceptance_rate:                              0.8399863117078281
#   log_density:                                  -80414.75981199446
#   hamiltonian_energy:                           80493.93146502542
#   hamiltonian_energy_error:                     0.1743696828634711
#   numerical_error:                              true
#   step_size:                                    0.0007395529164874169
#   nom_step_size:                                0.0007395529164874169
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00020852123351900516, 0. ...])
# ┌ Info: Finished 2000 sampling steps for 1 chains in 2898.0138857 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00020852123351900516, 0. ...]), kinetic=GaussianKinetic())
# │   κ = HMCKernel{AdvancedHMC.FullMomentumRefreshment, Trajectory{EndPointTS, Leapfrog{Float64}, FixedNSteps}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{EndPointTS}(integrator=Leapfrog(ϵ=0.00074), tc=FixedNSteps(50)))
# │   EBFMI_est = 0.12802787194800647
# └   average_acceptance_rate = 0.8502618668662216

# ESTIMATED ODE PARAMETER VALUES julia 
# > p2 = fhsamples1[2000][107]
# 1.661395306530168

# julia > p3 = fhsamples1[2000][108]
# 1.0657869663170956

# julia > p4 = fhsamples1[2000][109]
# 2.4584008298997286

# julia > p5 = fhsamples1[2000][110]
# 0.7798620351190229

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainlux1, dataset,
                                                              draw_samples = 2000,
                                                              l2std = [0.05, 0.05],
                                                              phystd = [0.05, 0.05],
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [
                                                                  Normal(1.5, 0.5),
                                                                  Normal(1.2, 0.5),
                                                                  Normal(3.3, 0.5),
                                                                  Normal(1.4, 0.5),
                                                              ], nchains = 2)

fh_mcmc_chain1, fhsamples1, fhstats1 = ahmc_bayesian_pinn_ode(prob, chainflux1, dataset,
                                                              draw_samples = 2000,
                                                              l2std = [0.05, 0.05],
                                                              phystd = [0.05, 0.05],
                                                              priorsNNw = (0.0, 3.0),
                                                              param = [
                                                                  Normal(1.5, 0.5),
                                                                  Normal(1.2, 0.5),
                                                                  Normal(3.3, 0.5),
                                                                  Normal(1.4, 0.5),
                                                              ], nchains = 2)

a = re(fhsamples1[1][2000][1:106])(time')
physsol3 = prob.u0 .+ (time' .- prob.tspan[1]) .* a
plot!(time, physsol3[1, :], label = "1882tanhlotka(0,10)1orderparra2000")
plot!(time, physsol3[2, :], label = "1882tanhlotka(0,10)1orderparra2000")
# ESTIMATED ODE PARAMETER VALUES
p2 = fhsamples1[1][2000][107]
p3 = fhsamples1[1][2000][108]
p4 = fhsamples1[1][2000][109]
p5 = fhsamples1[1][2000][110]
# ESTIMATED ODE PARAMETER VALUES
# julia > p2 = fhsamples1[1][2000][107]
# 1.6669602281639153

# julia > p3 = fhsamples1[1][2000][108]
# 1.0903160096758113

# julia > p4 = fhsamples1[1][2000][109]
# 2.238481154927218

# julia > p5 = fhsamples1[1][2000][110]
# 0.7268188330151024

a = re(fhsamples1[2][2000][1:106])(time')
physsol3 = prob.u0 .+ (time' .- prob.tspan[1]) .* a
plot!(time, physsol3[1, :], label = "1882tanhlotka(0,10)1orderparra-2-2000")
plot!(time, physsol3[2, :], label = "1882tanhlotka(0,10)1orderparra-2-2000")
# ESTIMATED ODE PARAMETER VALUES
p2 = fhsamples1[2][2000][107]
p3 = fhsamples1[2][2000][108]
p4 = fhsamples1[2][2000][109]
p5 = fhsamples1[2][2000][110]
# ESTIMATED ODE PARAMETER VALUES
# julia > p2 = fhsamples1[2][2000][107]
# 3.5186150758584023

# julia > p3 = fhsamples1[2][2000][108]
# 2.406806030552657

# julia > p4 = fhsamples1[2][2000][109]
# 0.40442930508264363

# julia > p5 = fhsamples1[2][2000][110]
# 0.1424727885253098

# Sampling 100%|███████████████████████████████| Time: 0:30:18
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    false
#   acceptance_rate:                              0.0
#   log_density:                                  -97606.98255431865
#   hamiltonian_energy:                           97659.11829068071
#   hamiltonian_energy_error:                     0.0
#   numerical_error:                              true
#   step_size:                                    0.0020032434261576465
#   nom_step_size:                                0.0020032434261576465
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00010366942053788189, 0. ...])
# Sampling 100%|███████████████████████████████| Time: 0:30:28
#   iterations:                                   2000
#   ratio_divergent_transitions:                  0.9
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      50
#   is_accept:                                    true
#   acceptance_rate:                              0.7771916449249778
#   log_density:                                  -115764.40131449902
#   hamiltonian_energy:                           115820.9235044525
#   hamiltonian_energy_error:                     0.25206831176183186
#   numerical_error:                              true
#   step_size:                                    0.0008138297363650001
#   nom_step_size:                                0.0008138297363650001
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([0.00011242546501281879, 0. ...])

# PROBLEM-3
linear = (u, p, t) -> -u / 5 + exp(-t + p[1] / 5) * cos(t) / p[2]
tspan = (0.0, 10.0)
u0 = 0.0
p = [2.0, 5.0]
prob = ODEProblem(linear, u0, tspan, p)

# PROBLEM-3
# linear = (u, p, t) -> -u / 5 + exp(-t / 5) * cos(t)
# linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
# tspan = (0.0f0, 10.0f0)
# u0 = 0.0f0
# prob = ODEProblem(ODEFunction(linear, analytic=linear_analytic), u0, tspan)

# PLOT SOLUTION AND CREATE DATASET
sol1 = solve(prob, Tsit5(); saveat = 0.1)
u = sol1.u[1:50]
time = sol1.t[1:50]
plot(sol1.t, sol1.u)

x̂ = collect(Float64, Array(u) + 0.005 * randn(size(u)))
dataset = [x̂, time]
plot!(time, x̂)

# --------------------------------------------------------------------------------------------

# NOTES (WILL CLEAR LATER)
# --------------------------------------------------------------------------------------------
# Hamiltonian energy must be lowest(more paramters the better is it to map onto them)
# full better than L2 and phy individual(test)
# in mergephys more points after training points is better from 20->40
# does consecutive runs bceome better? why?(plot 172)(same chain maybe)
# does density of points in timespan matter dataset vs internal timespan?(plot 172)(100+0.01)
# when training from 0-1 and phys from 1-5 with 1/150 simple nn slow,but bigger nn faster decrease in Hmailtonian
# bigger time interval more curves to adapt to only more parameters adapt to that, better NN architecture
# higher order logproblems solve better
# repl up up are same instances? but reexecute calls are new?

#Compare results against paper example
# Lux chains support (DONE)
# fix predictions for odes depending upon 1,p in f(u,p,t)(DONE)
# lotka volterra learn curve beyond l2 losses(L2 losses determine accuracy of parameters)(parameters cant run free ∴ L2 interval only)
# check if prameters estimation works(YES)
# lotka volterra parameters estimate (DONE)