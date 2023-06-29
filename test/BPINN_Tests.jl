# Testing out Code (will put in bpinntests.jl file in tests directory)

# Define ODE problem,conditions and Model,solve using NNODE.
using DifferentialEquations, MCMCChains
using NeuralPDE, Flux, OptimizationOptimisers

# prob1
linear = (u, p, t) -> -u / 5 + exp(-t / 5) * cos(t)
linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
# tspan = (0.0f0, 1.0f0)
tspan = (0.0f0, 10.0f0)
u0 = 0.0f0
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)

# prob2
# linear = (u, p, t) -> [cos(2pi * t)]
# tspan = (0.0f0, 1.0f0)
# u0 = [0.0f0]
# prob = ODEProblem(linear, u0, tspan)

chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
chain1 = chain

# Numerical and Analytical Solutions
ta = range(tspan[1], tspan[2], length = 100)
u = [linear_analytic(u0, nothing, ti) for ti in ta]
# sol1 = solve(prob, Tsit5())

# NNODE Solutions(500, 200 iters)
# opt = OptimizationOptimisers.Adam(0.1)
# sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose=true, abstol=1.0f-6, maxiters=500)
# sol2 = solve(prob, NeuralPDE.NNODE(chain, opt), verbose=true, abstol=1.0f-6, maxiters=200)

# initial NN predictions
# tspan = (0.0, 10.0)
# t = range(tspan[1], tspan[2], length=100)
time = vec(collect(Float64, ta))
# initialtry = vec(chain1(time'))

# BPINN AND TRAINING DATASET CREATION
x̂ = collect(Float64, Array(u) + 0.8 * randn(size(Array(u))))
# t = collect(Float32, ta)
dataset = (x̂, time)
typeof(dataset)
# chain1 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# parameters_initial, reconstruct = Flux.destructure(chain1)

using StatProfilerHTML, Profile
Profile.init() # returns the current settings
# # Profile.init(n=10^7, delay=0.1)

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

# x = get_params(chn)
# println(x)
# plot(chn["nnparameters[1]"]; size=(1200, 1200))
# names(chn)
# chn["nnparameters[1]"].data
# get(chn; section=:parameters)
# names(chn)
# sort(chn)
# get_params(chn)

# # testing stuff
# chain2 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn1 = bayesian_pinn_ode(prob, chain2, dataset; sampling_strategy=NUTS(0.9), num_samples=3000)
# chain3 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn2 = bayesian_pinn_ode(prob, chain3, dataset; sampling_strategy=NUTS(0.65), num_samples=3000)

# chain4 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn3 = bayesian_pinn_ode(prob, chain4, dataset; sampling_strategy=NUTS(0.9), num_samples=2000)
# chain5 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn4 = bayesian_pinn_ode(prob, chain5, dataset; sampling_strategy=NUTS(0.65), num_samples=2000)

# chain6 = Flux.Chain(Dense(1, 5, σ), Dense(5, 5, σ), Dense(5, 1))
# chn5 = bayesian_pinn_ode(prob, chain6, dataset; sampling_strategy=NUTS(0.65), num_samples=1000)
# chain7 = Flux.Chain(Dense(1, 5, σ), Dense(5, 5, σ), Dense(5, 1))
# chn6 = bayesian_pinn_ode(prob, chain7, dataset; sampling_strategy=NUTS(0.65), num_samples=2000)

# chain8 = Flux.Chain(Dense(1, 5, σ), Dense(5, 5, tanh), Dense(5, 1))
# chn7 = bayesian_pinn_ode(prob, chain8, dataset; sampling_strategy=NUTS(0.65), num_samples=1000)
# chain9 = Flux.Chain(Dense(1, 5, σ), Dense(5, 5, tanh), Dense(5, 1))
# chn8 = bayesian_pinn_ode(prob, chain9, dataset; sampling_strategy=NUTS(0.65), num_samples=2000)

# # PLOTTING CHAINS
# chn1
# chn2
# chn3
# chn4
# chn5
# chn6
# chn7
# chn8

# theta1 = MCMCChains.group(chn1, :nnparameters).value;
# theta2 = MCMCChains.group(chn2, :nnparameters).value;
# theta3 = MCMCChains.group(chn3, :nnparameters).value;
# theta4 = MCMCChains.group(chn4, :nnparameters).value;
# theta5 = MCMCChains.group(chn5, :nnparameters).value;
# theta6 = MCMCChains.group(chn6, :nnparameters).value;
# theta7 = MCMCChains.group(chn7, :nnparameters).value;
# theta8 = MCMCChains.group(chn8, :nnparameters).value;
# nn_forward(x, theta) = reconstruct(theta)(x)

# _, i = findmax(chn3[:lp])
# i = i.I[1]

# println(theta3[i, :])
# Z = nn_forward(dataset[2]', theta3[i, :])
# plot!(t, vec(Z))

# # ---------------------------------------------------------------
# @btime bayesian_pinn_ode(prob, chain1, dataset)

# chain10 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn9 = bayesian_pinn_ode(prob, chain10, dataset; sampling_strategy=NUTS(200, 0.9))
# chain11 = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn10 = bayesian_pinn_ode(prob, chain11, dataset; sampling_strategy=NUTS(200, 0.65))

# chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# chn = bayesian_pinn_ode(prob, chain, dataset)
# theta = MCMCChains.group(chn, :nnparameters).value;
# nn_forward(x, theta) = reconstruct(theta)(x)

# _, i = findmax(chn[:lp])
# i = i.I[1]
# Z = nn_forward(dataset[2]', theta[i, :])
# plot!(t, vec(Z), label="zamnnotime-old-old", legend=:bottomleft)
# # t and time are same

# plot(chn; colordim=:parameter)
# _, i = findmax(chn[:lp])
# print(chn[:])
# chn[:lp]
