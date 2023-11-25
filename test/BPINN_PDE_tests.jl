# using Test, MCMCChains, Lux, ModelingToolkit
# import ModelingToolkit: Interval, infimum, supremum
# using ForwardDiff, Distributions, OrdinaryDiffEq
# using Flux, AdvancedHMC, Statistics, Random, Functors
# using NeuralPDE, MonteCarloMeasurements
# using ComponentArrays

# # Forward solving example
# @parameters t
# @variables u(..)

# Dt = Differential(t)
# linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
# linear = (u, p, t) -> cos(2 * π * t)

# Dt = Differential(t)
# eqs = Dt(u(t)) - cos(2 * π * t) ~ 0
# bcs = [u(0) ~ 0.0]
# domains = [t ∈ Interval(0.0, 4.0)]

# chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1))
# init1, re1 = Flux.destructure(chainf)
# chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
# initl, st = Lux.setup(Random.default_rng(), chainl)

# @named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

# # non adaptive case
# discretization = NeuralPDE.PhysicsInformedNN([chainf], GridTraining([0.01]))
# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 1000,
#     bcstd = [0.02],
#     phystd = [0.01],
#     priorsNNw = (0.0, 1.0),
#     progress = true)

# discretization = NeuralPDE.PhysicsInformedNN([chainl], GridTraining([0.01]))
# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 1000,
#     bcstd = [0.02],
#     phystd = [0.01],
#     priorsNNw = (0.0, 1.0),
#     progress = true)

# tspan = (0.0, 4.0)
# t1 = collect(tspan[1]:0.01:tspan[2])

# out1 = re.([samples[i] for i in 800:1000])
# luxar1 = collect(out1[i](t1') for i in eachindex(out1))
# fluxmean = [mean(vcat(luxar1...)[:, i]) for i in eachindex(t1)]

# transsamples = [vector_to_parameters(sample, initl) for sample in samples]
# luxar2 = [chainl(t1', transsamples[i], st)[1] for i in 800:1000]
# luxmean = [mean(vcat(luxar2...)[:, i]) for i in eachindex(t1)]

# u = [linear_analytic(0, nothing, t) for t in t1]

# @test mean(abs.(u .- fluxmean)) < 5e-2
# @test mean(abs.(u .- luxmean)) < 5e-2

# # Parameter estimation example
# @parameters t p
# @variables u(..)

# Dt = Differential(t)
# # linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
# # linear = (u, p, t) -> cos(2 * π * t)
# eqs = Dt(u(t)) - cos(p * t) ~ 0
# bcs = [u(0.0) ~ 0.0]
# domains = [t ∈ Interval(0.0, 4.0)]

# p = 2 * π

# chainf = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 1))
# init1, re = Flux.destructure(chainf)
# chainl = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
# initl, st = Lux.setup(Random.default_rng(), chainl)

# # chainl([1, 2], initl, st)

# # using ComponentArrays
# # c = ComponentArrays.ComponentVector(a = [1, 2, 3], b = [1, 2, 3])

# # @parameters x y
# # @variables p(..) q(..) r(..) s(..)
# # Dx = Differential(x)
# # Dy = Differential(y)

# # # 2D PDE
# # eq = p(x) + q(y) + Dx(r(x, y)) + Dy(s(y, x)) ~ 0

# # # Initial and boundary conditions
# # bcs = [p(1) ~ 0.0f0, q(-1) ~ 0.0f0,
# #     r(x, -1) ~ 0.0f0, r(1, y) ~ 0.0f0,
# #     s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]

# # # Space and time domains
# # domains = [x ∈ Interval(0.0, 1.0),
# #     y ∈ Interval(0.0, 1.0)]

# # numhid = 3
# # chains = [[Lux.Chain(Lux.Dense(1, numhid, Lux.σ), Lux.Dense(numhid, numhid, Lux.σ),
# #     Lux.Dense(numhid, 1)) for i in 1:2]
# #     [Lux.Chain(Lux.Dense(2, numhid, Lux.σ), Lux.Dense(numhid, numhid, Lux.σ),
# #     Lux.Dense(numhid, 1)) for i in 1:2]]
# # discretization = NeuralPDE.PhysicsInformedNN(chains, QuadratureTraining())

# # @named pde_system = PDESystem(eq, bcs, domains, [x, y], [p(x), q(y), r(x, y), s(y, x)])

# # de = [:x, :y]
# # a[de[1]]
# # a = ComponentArrays.ComponentVector(x = 1)
# # a[:x]

# # pde_system.indvars
# # SymbolicUtils.istree(pde_system.depvars[3])
# # Symbolics.value(pde_system.depvars[3])

# @named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)], [p],
#     defaults = Dict(p => 3))

# ta = range(0.0, 4.0, length = 50)
# u = [linear_analytic(0.0, p, ti) for ti in ta]
# x̂ = collect(Float64, Array(u) + 0.2 .* Array(u) .* randn(size(u)))
# time = vec(collect(Float64, ta))
# # x = time .* 2.0
# dataset = [hcat(x̂, time), hcat(x̂, time), hcat(x̂, time, time), hcat(x̂, time, time)]
# hcat(datase[:, 2:end] for datase in dataset)

# discretization = NeuralPDE.PhysicsInformedNN([chainf],
#     GridTraining([0.01]),
#     param_estim = true)
# # println()

# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 1500, physdt = [1 / 20.0],
#     bcstd = [0.05],
#     phystd = [0.03], l2std = [0.02],
#     param = [Normal(9, 2)],
#     priorsNNw = (0.0, 10.0),
#     dataset = dataset,
#     progress = true)

# discretization = NeuralPDE.PhysicsInformedNN([chainl],
#     GridTraining(0.01),
#     param_estim = true)

# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 1500,
#     bcstd = [0.1],
#     phystd = [0.01], l2std = [0.01], param = [LogNormal(4, 2)],
#     priorsNNw = (0.0, 10.0),
#     dataset = dataset,
#     progress = true)

# tspan = (0.0, 4.0)
# t1 = collect(tspan[1]:0.01:tspan[2])

# out1 = re.([samples[i][1:(end - 1)] for i in 1300:1500])
# luxar1 = collect(out1[i](t1') for i in eachindex(out1))
# fluxmean = [mean(vcat(luxar1...)[:, i]) for i in eachindex(t1)]

# using Plots
# plotly()
# plot!(t1, fluxmean)
# plot!(dataset[1][:, 2], dataset[1][:, 1])

# samples[1500]
# samples[1500]

# transsamples = [vector_to_parameters(sample, initl) for sample[1:(end - 1)] in samples]
# luxar2 = [chainl(t1', transsamples[i], st)[1] for i in 1300:1500]
# luxmean = [mean(vcat(luxar2...)[:, i]) for i in eachindex(t1)]

# u = [linear_analytic(0, nothing, t) for t in t1]

# @test mean(abs.(u .- fluxmean)) < 5e-2
# @test mean(abs.(u .- luxmean)) < 5e-2

# @test mean(p .- [samples[i][end] for i in 1300:1500]) < 0.4 * p
# @test mean(p .- [samples[i][end] for i in 1300:1500]) < 0.4 * p

# plot(u)
# plot!(fluxmean)
# plot!(luxmean)

# using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
# import ModelingToolkit: Interval

# @parameters x y z
# @variables p(..) q(..) r(..) s(..)
# Dx = Differential(x)
# Dy = Differential(y)

# # 2D PDE
# eq = p(x) + z * q(y) + Dx(r(x, y)) + Dy(s(y, x)) ~ 0

# # Initial and boundary conditions
# bcs = [p(1) ~ 0.0f0, q(-1) ~ 0.0f0,
#     r(x, -1) ~ 0.0f0, r(1, y) ~ 0.0f0,
#     s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]

# # Space and time domains
# domains = [x ∈ Interval(0.0, 1.0),
#     y ∈ Interval(0.0, 1.0)]

# numhid = 3
# chains = [
#     Flux.Chain(Flux.Dense(1, numhid, σ), Flux.Dense(numhid, numhid, σ),
#         Flux.Dense(numhid, 1)),
#     Flux.Chain(Flux.Dense(1, numhid, σ), Flux.Dense(numhid, numhid, σ),
#         Flux.Dense(numhid, 1)),
#     Flux.Chain(Flux.Dense(2, numhid, σ), Flux.Dense(numhid, numhid, σ),
#         Flux.Dense(numhid, 1)),
#     Flux.Chain(Flux.Dense(2, numhid, σ), Flux.Dense(numhid, numhid, σ),
#         Flux.Dense(numhid, 1))]

# discretization = NeuralPDE.PhysicsInformedNN(chains,
#     GridTraining([0.1, 0.1]),
#     param_estim = true)
# discretization.strategy
# @named pde_system = PDESystem(eq,
#     bcs,
#     domains,
#     [x, y],
#     [p(x), q(y), r(x, y), s(y, x)],
#     [z],
#     defaults = Dict(z => 3))
# dataset = [hcat(x̂, time), hcat(x̂, time), hcat(x̂, time, time), hcat(x̂, time, time)]

# prob = SciMLBase.discretize(pde_system, discretization)
# a = [train_set[:, 2:end]' for train_set in dataset]
# b = zip(a)
# c = [yuh for yuh in b]
# c[[2], [1:50]]
# c[2]
# c[[2], [1:50]]
# zip(a)

# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 1500,
#     bcstd = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     phystd = [0.1], l2std = [0.1, 0.1, 0.1, 0.1],
#     priorsNNw = (0.0, 10.0),
#     param = [Normal(3, 2)],
#     dataset = dataset,
#     progress = true)
# # Example dataset structure
# matrix_dep_var_1 = dataset[1]

# matrix_dep_var_2 = dataset[2]

# dataset = [matrix_dep_var_1, matrix_dep_var_2]

# # Extract independent variable values
# indep_var_values = [matrix[:, 2:end] for matrix in dataset]

# # Adapt the existing code
# eltypeθ = Float64  # Replace with your desired element type
# pde_args = [[:indep_var]]

# # Generate training sets for each variable
# # Generate training sets for each variable
# pde_train_sets = map(pde_args) do bt
#     span = map(b -> vcat([indep_vars[:, b] for indep_vars in indep_var_values]...), bt)
#     _set = adapt(eltypeθ, hcat(span...))
# end

# pinnrep.depvars
# firstelement(domains[1])
# infimum(domains[1])
# infimum(domains[1].domain)
# domains = [x ∈ Interval(0.0, 1.0)]
# size(domains)

# callback = function (p, l)
#     println("Current loss is: $l")
#     return false
# end

# res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 100)

# # Paper experiments
# # function sir_ode!(u, p, t)
# #     (S, I, R) = u
# #     (β, γ) = p
# #     N = S + I + R

# #     dS = -β * I / N * S
# #     dI = β * I / N * S - γ * I
# #     dR = γ * I
# #     return [dS, dI, dR]
# # end;

# # δt = 1.0
# # tmax = 40.0
# # tspan = (0.0, tmax)
# # u0 = [990.0, 10.0, 0.0]; # S,I,R
# # p = [0.5, 0.25]; # β,γ (removed c as a parameter as it was just getting multipled with β, so ideal value for c and β taken in new ideal β value)
# # prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
# # sol = solve(prob_ode, Tsit5(), saveat = δt / 5)
# # sig = 0.20
# # data = Array(sol)
# # dataset = [
# #     data[1, :] .+ (minimum(data[1, :]) * sig .* rand(length(sol.t))),
# #     data[2, :] .+ (mean(data[2, :]) * sig .* rand(length(sol.t))),
# #     data[3, :] .+ (mean(data[3, :]) * sig .* rand(length(sol.t))),
# #     sol.t,
# # ]
# # priors = [Normal(1.0, 1.0), Normal(0.5, 1.0)]

# # plot(sol.t, dataset[1], label = "noisy s")
# # plot!(sol.t, dataset[2], label = "noisy i")
# # plot!(sol.t, dataset[3], label = "noisy r")
# # plot!(sol, labels = ["s" "i" "r"])

# # chain = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 8, tanh),
# #     Flux.Dense(8, 3))

# # Adaptorkwargs = (Adaptor = AdvancedHMC.StanHMCAdaptor,
# #     Metric = AdvancedHMC.DiagEuclideanMetric, targetacceptancerate = 0.8)

# # alg = BNNODE(chain;
# #     dataset = dataset,
# #     draw_samples = 500,
# #     l2std = [5.0, 5.0, 10.0],
# #     phystd = [1.0, 1.0, 1.0],
# #     priorsNNw = (0.01, 3.0),
# #     Adaptorkwargs = Adaptorkwargs,
# #     param = priors, progress = true)

# # # our version
# # @time sol_pestim3 = solve(prob_ode, alg; estim_collocate = true, saveat = δt)
# # @show sol_pestim3.estimated_ode_params

# # # old version
# # @time sol_pestim4 = solve(prob_ode, alg; saveat = δt)
# # @show sol_pestim4.estimated_ode_params

# # # plotting solutions
# # plot(sol_pestim3.ensemblesol[1], label = "estimated x1")
# # plot!(sol_pestim3.ensemblesol[2], label = "estimated y1")
# # plot!(sol_pestim3.ensemblesol[3], label = "estimated z1")

# # plot(sol_pestim4.ensemblesol[1], label = "estimated x2_1")
# # plot!(sol_pestim4.ensemblesol[2], label = "estimated y2_1")
# # plot!(sol_pestim4.ensemblesol[3], label = "estimated z2_1")

# # discretization = NeuralPDE.PhysicsInformedNN(chainf,
# #     GridTraining([
# #         0.01,
# #     ]),
# #     adaptive_loss = MiniMaxAdaptiveLoss(2;
# #         pde_max_optimiser = Flux.ADAM(1e-4),
# #         bc_max_optimiser = Flux.ADAM(0.5),
# #         pde_loss_weights = 1,
# #         bc_loss_weights = 1,
# #         additional_loss_weights = 1)

# #     # GradientScaleAdaptiveLoss{Float64, ForwardDiff.Dual{Float64}}(2;
# #     #     weight_change_inertia = 0.9,
# #     #     pde_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# #     #         ntuple(_ -> 0.0, 19)),
# #     #     bc_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# #     #         ntuple(_ -> 0.0, 19)),
# #     #     additional_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# #     #         ntuple(_ -> 0.0, 19)))
# #     #     GradientScaleAdaptiveLoss{Float64, ForwardDiff.Dual{Float64, Float64}}(2;
# #     # weight_change_inertia = 0.9,
# #     # pde_loss_weights = ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19)),
# #     # bc_loss_weights = ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19)),
# #     # additional_loss_weights = ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))))
# # )

# # # ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))
# # # typeof(ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))) <: ForwardDiff.Dual

# # a = GradientScaleAdaptiveLoss{Float64, ForwardDiff.Dual{Float64, Float64}}(2;
# # weight_change_inertia = 0.9,
# # pde_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# # zeros(19))),
# # bc_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# # zeros(19)),
# # additional_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# #     zeros(19))

# # as = ForwardDiff.Dual{Float64}(1.0, ForwardDiff.Partials(ntuple(_ -> 0.0, 19)))
# # typeof(ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19)))

# # a = GradientScaleAdaptiveLoss{Float64, Float64}(2; weight_change_inertia = 0.9,
# #     pde_loss_weights = 1,
# #     bc_loss_weights = 1,
# #     additional_loss_weights = 1)
# # ForwardDiff.Dual{Float64, Float64, 19} <: ForwardDiff.Dual{Float64, Float64}
# # typeof(ntuple(_ -> 0.0, 19)) <: Tuple
# # ForwardDiff.Dual{Float64}(ForwardDiff.value(1.0), ntuple(_ -> 0.0, 19))
# # typeof(ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))) <: Real
