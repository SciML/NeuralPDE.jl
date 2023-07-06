# Testing out Code (will put in bpinntests.jl file in tests directory)

# Define ODE problem,conditions and Model,solve using NNODE.
using DifferentialEquations, MCMCChains, ForwardDiff
using NeuralPDE, Flux, OptimizationOptimisers
using StatProfilerHTML, Profile, Statistics
using BenchmarkTools, Plots, StatsPlots
plotly()
Profile.init()   # returns the current settings
# # Profile.init(n=10^7, delay=0.1)

# # # prob1
# linear = (u, p, t) -> -u / 5 + exp(-t / 5) * cos(t)
# linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
# tspan = (0.0f0, 10.0f0)
# u0 = 0.0f0
# typeof(u0)
# typeof(tspan[1])
# prob = ODEProblem(ODEFunction(linear, analytic=linear_analytic), u0, tspan)
# typeof(linear(u0, prob.p, tspan[1]))
# prob = ODEProblem(ODEFunction(linear, analytic=linear_analytic), u0, tspan)

# using activation funciton on first layer signficantly improved stuuff,1 for none,2 for sigmoid,3 for tanh
# for above tspan(0-10)
# graph 4 is tspan(0-2) sigmoid, 5 for tanh,6 for none
# graph 7 is for all sampled results for graph 4 Case,graph 8 is the chain got
# graph 9 is for all sampled results for graph 5 Case,graph 10 is the chain got
# correlation between activation function,sigmoid,tanh clearly better
# number of timepoints to train on clearly affects results
# all above are for 100 datapoints

# graph 11 is tanh, for timespan(0-1), 20 datapoints, graph 12 for all samples,graph 13 is its chain
# graph 14 is sigmoid, for timespan(0-1), 20 datapoints, graph 15 for all samples,graph 16 is its chain

# prob2
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)
# tspan = (0.0, 10.0)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# Numerical and Analytical Solutions
ta = range(tspan[1], tspan[2], length = 20)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
sol1 = solve(prob, Tsit5())
plot(sol1.t, sol1.u)

# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x̂ = collect(Float64, Array(u) + 0.8 * randn(size(u)))
time = vec(collect(Float64, ta))
dataset = (x̂, time)

# Call BPINN, create chain
chainfh = Flux.Chain(Dense(1, 5, sigmoid), Dense(5, 1))
fhsamples, fhstats = ahmc_bayesian_pinn_ode(prob, chainfh, dataset, warmup_samples = 500,
                                            draw_samples = 100)
init, re = destructure(chainfh)

# plotting time points
t = time
p = prob.p

# PLOTTING MEDIAN,MEANS AND iTH PARAMETER CURVES
# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

# Create mcmc chain
samples = fhsamples
matrix_samples = hcat(samples...)
fh_mcmc_chain = Chains(matrix_samples')  # Create a chain from the reshaped samples

means = mean(matrix_samples, dims = 2)
medians = median(matrix_samples, dims = 2)

# plotting average of final nn outputs
out = re.(fhsamples)
yu = collect(out[i](t') for i in eachindex(out))
yu = vcat(yu...)
a = [mean(yu[:, i]) for i in eachindex(t)]
plot!(t, prob.u0 .+ (t .- prob.tspan[1]) .* a, label = "curve averages")

# plotting i'th sampled parameters NN output
a = vec(re(vec(fhsamples[100]))(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3, label = "y(x,t) 100th curve")

# plotting curve when using mean of sampled parameters
a = vec(re(vec(means))(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3, label = "y(x,t) means curve")

# plotting curve when using median of sampled parameters
a = vec(re(vec(medians))(t'))
physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* a
plot!(t, physsol3, label = "y(x,t) median curve")

# ALL SAMPLES PLOTS--------------------------------------
# Plot problem and its solution
plot(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)
physsol1 = [linear_analytic(prob.u0, p, t[i]) for i in eachindex(t)]
physsol2 = [linear(physsol1[i], p, t[i]) for i in eachindex(t)]
plot!(t, physsol1, label = "y(x,t)")
plot!(t, physsol2, label = "y'(x,t)")

# Plot each iv'th sample in samples
function realsol(p, out, linear, t, iv)
    outpreds = out(t')
    physsol3 = prob.u0 .+ (t .- prob.tspan[1]) .* vec(outpreds)
    # physsol4 = [linear(physsol3[i], p, t[i]) for i in eachindex(t)]
    plot!(t, physsol3, label = "y(x,t) $iv th curve")
    # plot!(t, physsol4, label="y'(x,t) $iv th curve")
end

# Plot all NN parameter samples curve posibilities
for i in eachindex(fhsamples)
    out = re(fhsamples[i])
    realsol(p, out, linear, t, i)
end

# added this as above plot takes time this lower line renders the above plot faster
plot!(title = "Problem1 y'(x,t),y(x,t) for ODE,BPINN", legend = :outerbottomright)

# PLOT MCMC Chain
plot(fh_mcmc_chain)
# -------------------------------------------------------------------------------------------------------------------------
# (Above is the testing code as commented,use it in another env in case package issues arise)

# -------------------------------------------------------------------------------------------------------------------------

# NOTE:bayesian_pinn_ode for 500 points and 100 samples took >30mins and inaccurate results
# also default bayesian_pinn_ode call takes 3hrs ish

# Below calls with stats are usign AdvancedHMC.jl

nsamples, nstats = ahmc_bayesian_pinn_ode(prob,
                                          chain1,
                                          dataset,
                                          draw_samples = 10,
                                          warmup_samples = 10)
# (100 points)
# Sampling 100%|███████████████████████████████| Time: 0:00:05
#   iterations:                                   10
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.2
#   n_steps:                                      31
#   is_accept:                                    true
#   acceptance_rate:                              0.7370448630361337
#   log_density:                                  -15314.854960676601
#   hamiltonian_energy:                           15359.332549348916
#   hamiltonian_energy_error:                     -0.9167791773706995
#   max_hamiltonian_energy_error:                 1.3162162828957662
#   tree_depth:                                   5
#   numerical_error:                              false
#   step_size:                                    0.006886482653902681
#   nom_step_size:                                0.006886482653902681
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...])
# ┌ Info: Finished 10 sampling steps for 1 chains in 5.4824978 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00534), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0))) 
# │   EBFMI_est = 1.0272402712636524
# └   average_acceptance_rate = 0.7238602930664688

nsamples1, nstats1 = ahmc_bayesian_pinn_ode(prob,
                                            chain1,
                                            dataset,
                                            draw_samples = 100,
                                            warmup_samples = 500)
# 100 points at (100,500)
# Sampling 100%|███████████████████████████████| Time: 0:13:19
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.9515248815726415
#   log_density:                                  -15314.4778127203
#   hamiltonian_energy:                           15318.692966029752
#   hamiltonian_energy_error:                     0.04905899994082574
#   max_hamiltonian_energy_error:                 0.16138963089179015
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.003125217749335249
#   nom_step_size:                                0.003125217749335249
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([50.116433994066654, 15.173 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 799.2808691 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([50.116433994066654, 15.173 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00417), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0))) 
# │   EBFMI_est = 0.764991414813652
# └   average_acceptance_rate = 0.7851066384751426

vecsamples, vecstats = ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples = 100)
# 100 points (100,1000)run
# Sampling 100%|███████████████████████████████| Time: 0:10:16
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.13
#   n_steps:                                      7
#   is_accept:                                    true
#   acceptance_rate:                              0.09581456846157546
#   log_density:                                  -15320.697538679076
#   hamiltonian_energy:                           15327.19997184089
#   hamiltonian_energy_error:                     0.4134248195987311
#   max_hamiltonian_energy_error:                 1088.4832929759013
#   tree_depth:                                   2
#   numerical_error:                              true
#   step_size:                                    0.011374348521550458
#   nom_step_size:                                0.011374348521550458
#   is_adapt:                                     true

@profview ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples = 100)
# 100 points (100,1000)run
# Sampling 100%|███████████████████████████████| Time: 0:16:05
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.08
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.9858858183628039
#   log_density:                                  -15321.012038470617
#   hamiltonian_energy:                           15327.267686544108
#   hamiltonian_energy_error:                     0.035219259316363605
#   max_hamiltonian_energy_error:                 0.04619328611624951
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.00375498199774842
#   nom_step_size:                                0.00375498199774842
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([7.186515813469459, 49.1935 ...])

nextsamples = ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples = 100)
# 100 points (100,500) run
# Sampling 100%|███████████████████████████████| Time: 0:11:48
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.12
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.9495408070146313
#   log_density:                                  -15313.563215779219
#   hamiltonian_energy:                           15321.769836504141
#   hamiltonian_energy_error:                     -0.03376652941733482
#   max_hamiltonian_energy_error:                 0.14091616958830855
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.002645824472237061
#   nom_step_size:                                0.002645824472237061
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([12.368212100200626, 0.0005 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 708.490331 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([12.368212100200626, 0.0005 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00352), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.513799978316982
# └   average_acceptance_rate = 0.7841746849813168

# 500 points gang
ta = range(tspan[1], tspan[2], length = 500)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
time = vec(collect(Float64, ta))
x̂ = collect(Float64, Array(u) + 0.8 * randn(size(Array(u))))
dataset = (x̂, time)

# 500 points
next500pointssamples = ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples = 100)
next500pointssamples1 = ahmc_bayesian_pinn_ode(prob, chain1, dataset)
next500pointssamples2 = ahmc_bayesian_pinn_ode(prob,
                                               chain1,
                                               dataset,
                                               draw_samples = 100,
                                               warmup_samples = 1000)

# @profilehtml chn1 = bayesian_pinn_ode(prob, chain1, dataset, num_samples=100)
# @profilehtml chn1 = bayesian_pinn_ode(prob, chain1, dataset, num_samples=500)
# Profile.init(n=10^7, delay=0.01)
# @profilehtml chn1 = bayesian_pinn_ode(prob, chain1, dataset, num_samples=500)

chn1 = bayesian_pinn_ode(prob, chain1, dataset, num_samples = 500)
ta = range(tspan[1], tspan[2], length = 100)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
time = vec(collect(Float32, ta))
x̂ = collect(Float32, Array(u) .+ 0.8 .* randn(size(Array(u))))
dataset = (x̂, time)
@profilehtml chn2 = bayesian_pinn_ode(prob, chain1, dataset, num_samples = 100)

chn = bayesian_pinn_ode(prob, chain1, dataset, num_samples = 2000)
chn1
theta = MCMCChains.group(chn1, :nnparameters).value;
theta
nn_forward(x, theta) = reconstruct(theta)(x)
_, i = findmax(chn[:lp])
i = i.I[1]
aftertrain = vec(chain1(time'))
Z = nn_forward(dataset[2]', theta[i, :])

# Reproduce 3 hr slow sampling
# dataset would be (x̂,t) for this example size = (5,5)
# # priors: pdf for W,b
# using Turing, LinearAlgebra
# function genpoints(num)
#     x = rand(num)
#     t = rand(num)
#     return (x, t)
# end

# function bayesian_pinn!(chain::Flux.Chain,
#     dataset::Tuple{Vector{Float64},Vector{Float64}};
#     sampling_strategy=NUTS(0.65), num_samples=1000)
#     nnparameters, recon = Flux.destructure(chain)
#     nparameters = length(nnparameters)
#     sig = sqrt(1.0 / 0.09)

#     Turing.@model function bayes_pinn(dataset::Tuple{Vector{Float64},Vector{Float64}})
#         # priors for NN parameters(not included bias yet?) - P(Θ)
#         nnparameters ~ MvNormal(zeros(nparameters), Diagonal(sig * ones(nparameters)))

#         nn = recon(nnparameters)
#         preds = nn(dataset[2]')

#         # this would be solutions for data(μ) vs NN preds(physsol)
#         μ = genpoints(500)[1] + 0.8 * randn(500)
#         physsol = genpoints(500)[1]    # (physsol involves numerical derivative at each timepoint.)
#         # both these above are got seperately via diff function calls

#         # likelihood for NN pred(physsol) vs data(μ) - P(phys | Θ)
#         if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
#             Turing.@addlogprob! loglikelihood(MvNormal(μ, Diagonal(sig .* ones(length(μ)))), physsol)
#         end

#         # likelihood for dataset vs NN pred  - P( X̄ | Θ)
#         dataset[1] ~ MvNormal(vec(preds), sig .* ones(length(dataset[2])))
#     end

#     model = bayes_pinn(dataset)
#     ch = sample(model, sampling_strategy, num_samples)
#     return ch
# end
# datapoints = genpoints(500)
# chain0 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))

# chn1 = bayesian_pinn(chain0, datapoints, num_samples=1000)

# plotting solutions
using Plots, StatsPlots
plotly()
# Graph's Title and init
plot(title = "exp(-T/5)*sin(T) (0->10)", legend = :bottomright)
plot!(ta, u, label = "Analytic", legend = :topright)
plot!(sol1, label = "Numerical", xlabel = "t", ylabel = "u")
plot!(sol.t, sol.u, label = "NNODE_500iters")
plot!(sol2.t, sol2.u, label = "NNODE_200iters")
plot!(time, initialtry, label = "initial NN output")
plot(chn)
plot!(time, aftertrain, label = "final NN output", legend = :topleft)
plot!(time, vec(Z), label = "BPINNpreds-2", legend = :bottomright)

# # t and time are same

# PLOTTING FUNCTIONS
using Plots
plotly()
plot!(title = "exp(-T/5)*sin(T) (0->10)logformpcforwa with var bnn ")
ch = reconstruct(logformsamples[20])
plot!(dataset[2], vec(ch(dataset[2]')), label = "100 th output")
function bruh(samples)
    plot(title = "exp(-T/5)*sin(T) (0->10)", legend = :bottomright)
    for i in eachindex(samples)
        ch = reconstruct(samples[i])
        plot!(dataset[2], vec(ch(dataset[2]')), label = "$i th output")
    end
end

logpdfsamples, logpdfstats = ahmc_bayesian_pinn_ode(prob, chain1, dataset,
                                                    draw_samples = 100)
logpdfsamples
# 100 points 1000warmup,100drawn samples
# RUN - 1
# Sampling 100%|███████████████████████████████| Time: 0:12:50
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.14
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.9563144470142292
#   log_density:                                  -15271.845533849797
#   hamiltonian_energy:                           15278.121731738258
#   hamiltonian_energy_error:                     0.12927246448271035
#   max_hamiltonian_energy_error:                 -0.2010435676365887
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.0054621222715614615
#   nom_step_size:                                0.0054621222715614615
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.0003076056962720712, 5.3 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 770.4964452 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.0003076056962720712, 5.3 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00733), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.34253160077428535
# └   average_acceptance_rate = 0.7882127810769348

# RUN - 2
# Sampling 100%|███████████████████████████████| Time: 0:09:20
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.1
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.8768665862746476
#   log_density:                                  -15272.886204006449
#   hamiltonian_energy:                           15281.696956530182
#   hamiltonian_energy_error:                     0.24551082678044622
#   max_hamiltonian_energy_error:                 0.2773981831378478
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.004382810950719518
#   nom_step_size:                                0.004382810950719518
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([34.912048543892865, 35.633 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 560.851299 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([34.912048543892865, 35.633 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.0051), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.40267051146009725
# └   average_acceptance_rate = 0.7862124562827073
bruh(logpdfsamples)

loglikesamples, loglikestats = ahmc_bayesian_pinn_ode(prob, chain1, dataset,
                                                      draw_samples = 100)
loglikesamples
# 100 points 1000warmup,100drawn samples
# RUN-1
# Sampling 100%|███████████████████████████████| Time: 0:09:45
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.13
#   n_steps:                                      4
#   is_accept:                                    true
#   acceptance_rate:                              0.030908461126574967
#   log_density:                                  -15271.887943885158
#   hamiltonian_energy:                           15283.930415289895
#   hamiltonian_energy_error:                     0.0
#   max_hamiltonian_energy_error:                 1128.8259864720458
#   tree_depth:                                   2
#   numerical_error:                              true
#   step_size:                                    0.008867437749359755
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([2.88135620435816, 17.49607 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 585.0805829 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([2.88135620435816, 17.49607 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00221), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.37259664441661766
# └   average_acceptance_rate = 0.781612814236198

# RUN-2
# Sampling 100%|███████████████████████████████| Time: 0:08:09
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.15
#   n_steps:                                      59
#   is_accept:                                    true
#   acceptance_rate:                              0.35444357264885873
#   log_density:                                  -15271.845139989402
#   hamiltonian_energy:                           15276.51803860688
#   hamiltonian_energy_error:                     0.0869020819063735
#   max_hamiltonian_energy_error:                 1243.6540616378516
#   tree_depth:                                   5
#   numerical_error:                              true
#   step_size:                                    0.010087308582972208
#   nom_step_size:                                0.010087308582972208
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([5.8417644509391105, 21.300 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 489.3605312 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([5.8417644509391105, 21.300 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00452), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.40053207226751153
# └   average_acceptance_rate = 0.7855541665544353
bruh(loglikesamples)

logformsamples, logformstats = ahmc_bayesian_pinn_ode(prob, chain1, dataset,
                                                      draw_samples = 100)
logformsamples
# -------> return sum(abs2, (nnsol .- physsol) ./ (-2 * (var^2)))
# 100 points 1000warmup,100drawn samples
# RUN-1
# Sampling 100%|███████████████████████████████| Time: 0:00:06
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.04
#   n_steps:                                      3
#   is_accept:                                    true
#   acceptance_rate:                              0.9936457746418542
#   log_density:                                  16371.958270823849
#   hamiltonian_energy:                           -16353.795618226019
#   hamiltonian_energy_error:                     0.0031999390394048532
#   max_hamiltonian_energy_error:                 0.012711568355371128
#   tree_depth:                                   2
#   numerical_error:                              false
#   step_size:                                    0.001013347477860843
#   nom_step_size:                                0.001013347477860843
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.00023705855136240276, 0. ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 6.5440993 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00023705855136240276, 0. ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00147), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.003261088919982054
# └   average_acceptance_rate = 0.7717655246027548

# RUN-2
# Sampling 100%|███████████████████████████████| Time: 0:00:05
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.02
#   n_steps:                                      3
#   is_accept:                                    true
#   acceptance_rate:                              0.9331758601037174
#   log_density:                                  24029.267730447515
#   hamiltonian_energy:                           -23793.7201448559
#   hamiltonian_energy_error:                     0.14239990540227154
#   max_hamiltonian_energy_error:                 0.14239990540227154
#   tree_depth:                                   2
#   numerical_error:                              false
#   step_size:                                    0.0015450783603555866
#   nom_step_size:                                0.0015450783603555866
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.0001884855949168989, 0.0 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 5.8228399 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.0001884855949168989, 0.0 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00201), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.002734871196746394
# └   average_acceptance_rate = 0.7742027133641893

# USING FORWARDDIFF RUN-1
# Sampling 100%|███████████████████████████████| Time: 0:07:26
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.03
#   n_steps:                                      63
#   is_accept:                                    true
#   acceptance_rate:                              0.0901623955375686
#   log_density:                                  33531.225900141486
#   hamiltonian_energy:                           -31905.391124649912
#   hamiltonian_energy_error:                     1.0579141952184727
#   max_hamiltonian_energy_error:                 470.8753844133971
#   tree_depth:                                   6
#   numerical_error:                              false
#   step_size:                                    0.00043740944337070624
#   nom_step_size:                                0.00043740944337070624
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.00016850576931927736, 0. ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 446.7712559 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00016850576931927736, 0. ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.000122), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.007014130445885143
# └   average_acceptance_rate = 0.774046980073258

# USING FORWARDDIFF RUN-2
# Sampling 100%|███████████████████████████████| Time: 0:06:00
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.02
#   n_steps:                                      31
#   is_accept:                                    true
#   acceptance_rate:                              0.12297576613170554
#   log_density:                                  22100.349958875497
#   hamiltonian_energy:                           -22091.533249050637
#   hamiltonian_energy_error:                     0.0
#   max_hamiltonian_energy_error:                 28.814220876411127
#   tree_depth:                                   5
#   numerical_error:                              false
#   step_size:                                    0.0005074527614074749
#   nom_step_size:                                0.0005074527614074749
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.00016802275586959888, 0. ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 360.606352 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.00016802275586959888, 0. ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00015), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.003456680349189868
# └   average_acceptance_rate = 0.774810796422875

# USING ONLY L2LOSS + FORWARDDIFF
# julia> logformsamples, logformstats = ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples=100)
# Sampling 100%|███████████████████████████████| Time: 0:00:24
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      959
#   is_accept:                                    true
#   acceptance_rate:                              0.6683146173012535
#   log_density:                                  -3340.3826246506915
#   hamiltonian_energy:                           3346.9492811468526
#   hamiltonian_energy_error:                     0.021912054422955407
#   max_hamiltonian_energy_error:                 366.35912006145463
#   tree_depth:                                   9
#   numerical_error:                              false
#   step_size:                                    0.0029268197636419338
#   nom_step_size:                                0.0029268197636419338
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.0012634344496764663, 0.2 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 24.9109321 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.0012634344496764663, 0.2 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.00234), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.05611701717517014
# └   average_acceptance_rate = 0.7781185997585821

# julia> logformsamples, logformstats = ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples=100)
# Sampling 100%|███████████████████████████████| Time: 0:00:17
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.15
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.9999169847480434
#   log_density:                                  -3080.397360098603
#   hamiltonian_energy:                           3083.111687996316
#   hamiltonian_energy_error:                     -0.0951502568759679
#   max_hamiltonian_energy_error:                 -0.16819624547269996
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.00044361575140609746
#   nom_step_size:                                0.00044361575140609746
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([0.008488990018137863, 0.04 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 17.2757482 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([0.008488990018137863, 0.04 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.000653), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.10017704954954774
# └   average_acceptance_rate = 0.7711002361182879
bruh(logformsamples)

# -------------->WORKING ON ACCURACY(START WITH BNN(L2LOSS ONLY))
# PLOTTING FUNCTIONS
# VARIANCE affects answers
plot!(title = "exp(-T/5)*sin(T) (0->10)BNN Logpdf 1var ")
ch = reconstruct(ffsamples[1])
plot!(dataset[2], vec(ch(dataset[2]')), label = "100 th output")
function bruh(samples)
    plot(title = "exp(-T/5)*sin(T) (0->10)", legend = :bottomright)
    for i in eachindex(samples)
        ch = reconstruct(samples[i])
        plot!(dataset[2], vec(ch(dataset[2]')), label = "$i th output")
    end
end
using Statistics

var(dataset[2])

# COMPLETE LOGPDF FORMULA(logpdf of mvnormal,0.64 var)
ffsamples, ffstats = ahmc_bayesian_pinn_ode(prob, chain1, dataset, draw_samples = 100)
# RUN-1
# Sampling 100%|███████████████████████████████| Time: 0:00:23
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.09
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.992595083092237
#   log_density:                                  -118.52948000023393
#   hamiltonian_energy:                           127.74915583867521
#   hamiltonian_energy_error:                     0.009264370731486338
#   max_hamiltonian_energy_error:                 -0.07638609599329982
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.04027281953328647
#   nom_step_size:                                0.04027281953328647
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([160.43539397075622, 2130.1 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 23.8526909 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([160.43539397075622, 2130.1 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.0578), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 1.4217689218482938
# └   average_acceptance_rate = 0.78812861517715

# RUN-2
# Sampling 100%|███████████████████████████████| Time: 0:00:20
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.12
#   n_steps:                                      12
#   is_accept:                                    true
#   acceptance_rate:                              0.2474803047972032
#   log_density:                                  -119.92078012420382
#   hamiltonian_energy:                           125.75567944951601
#   hamiltonian_energy_error:                     -0.9345980374540943
#   max_hamiltonian_energy_error:                 1596.4982651510013
#   tree_depth:                                   3
#   numerical_error:                              true
#   step_size:                                    0.0960487466692949
#   nom_step_size:                                0.0960487466692949
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([388.0534809193257, 1116.95 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 20.0106955 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([388.0534809193257, 1116.95 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.0356), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 0.41801982165121526
# └   average_acceptance_rate = 0.7778449909948125

# COMPLETE LOGPDF FORMULA(logpdf of mvnormal,0.09 alpha)
# RUN - 1
# Sampling 100%|███████████████████████████████| Time: 0:00:20
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.13
#   n_steps:                                      6
#   is_accept:                                    true
#   acceptance_rate:                              0.007879101773356835
#   log_density:                                  -166.0611415869184
#   hamiltonian_energy:                           177.6335770491318
#   hamiltonian_energy_error:                     3.263792609192734
#   max_hamiltonian_energy_error:                 1947.0817156242097
#   tree_depth:                                   2
#   numerical_error:                              true
#   step_size:                                    0.2917869566189766
#   nom_step_size:                                0.2917869566189766
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([30386.033908257898, 5329.2 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 20.8084753 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([30386.033908257898, 5329.2 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.0699), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 2.028980396225055
# └   average_acceptance_rate = 0.7777386701692393

# RUN - 2
# Sampling 100%|███████████████████████████████| Time: 0:00:20
#   iterations:                                   100
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.12
#   n_steps:                                      651
#   is_accept:                                    true
#   acceptance_rate:                              0.6379211896489467
#   log_density:                                  -164.08209791645453
#   hamiltonian_energy:                           170.11216603261437
#   hamiltonian_energy_error:                     0.07380375065747558
#   max_hamiltonian_energy_error:                 8300.704104737693
#   tree_depth:                                   9
#   numerical_error:                              true
#   step_size:                                    0.2431568484312491
#   nom_step_size:                                0.2431568484312491
#   is_adapt:                                     true
#   mass_matrix:                                  DiagEuclideanMetric([6861.381492397943, 9648.55 ...])
# ┌ Info: Finished 100 sampling steps for 1 chains in 20.4012424 (s)
# │   h = Hamiltonian(metric=DiagEuclideanMetric([6861.381492397943, 9648.55 ...]), kinetic=AdvancedHMC.GaussianKinetic())
# │   κ = AdvancedHMC.HMCKernel{AdvancedHMC.FullMomentumRefreshment, AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHMC.Leapfrog{Float64}, AdvancedHMC.GeneralisedNoUTurn{Float64}}}(AdvancedHMC.FullMomentumRefreshment(), Trajectory{AdvancedHMC.MultinomialTS}(integrator=Leapfrog(ϵ=0.183), tc=AdvancedHMC.GeneralisedNoUTurn{Float64}(10, 1000.0)))
# │   EBFMI_est = 1.9409429547327703
# └   average_acceptance_rate = 0.7830404564067538
bruh(ffsamples)
