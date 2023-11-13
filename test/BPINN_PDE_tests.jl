# ______________________________________________PDE_BPINN_SOLVER_________________________________________________________________

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
using Plots, OrdinaryDiffEq, Distributions, Random
import ModelingToolkit: Interval, infimum, supremum
# # Testing Code
using Test, MCMCChains
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, OptimizationOptimisers, AdvancedHMC, Lux
using Statistics, Random, Functors, ComponentArrays
using NeuralPDE, MonteCarloMeasurements
# @parameters t
# @variables x(..) y(..)

# α, β, γ, δ = [1.5, 1.0, 3.0, 1.0]

# Dt = Differential(t)
# eqs = [Dt(x(t)) ~ (α - β * y(t)) * x(t), Dt(y(t)) ~ (δ * x(t) - γ) * y(t)]
# bcs = [x(0) ~ 1.0, y(0) ~ 1.0]
# domains = [t ∈ Interval(0.0, 6.0)]

# chain = [Flux.Chain(Flux.Dense(1, 4, tanh), Flux.Dense(4, 4),
#         Flux.Dense(4, 1)), Flux.Chain(Flux.Dense(1, 4, tanh), Flux.Dense(4, 4),
#         Flux.Dense(4, 1))]

# chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh),
#     Flux.Dense(6, 2))

# init, re = destructure(chainf)
# init1, re1 = destructure(chain[1])
# init2, re2 = destructure(chain[2])
# chainf = re(ones(size(init)))
# chain[1] = re1(ones(size(init1)))
# chain[2] = re2(ones(size(init2)))

# discretization = NeuralPDE.PhysicsInformedNN(chain,
#     GridTraining([0.01]))
# @named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t)])

# mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system, discretization;
#     draw_samples = 100,
#     bcstd = [0.1, 0.1],
#     phystd = [0.1, 0.1], priorsNNw = (0.0, 10.0), progress = true)

# # FLUX CHAIN post sampling
# tspan = (0.0, 6.0)
# t1 = collect(tspan[1]:0.01:tspan[2])
# out1 = re1.([samples[i][1:33]
#              for i in 80:100])
# out2 = re2.([samples[i][34:end]
#              for i in 80:100])

# luxar1 = collect(out1[i](t1') for i in eachindex(out1))
# luxar2 = collect(out2[i](t1') for i in eachindex(out2))
# plot(t1, luxar1[end]')
# plot!(t1, luxar2[end]')

# # LUX CHAIN post sampling
# θinit, st = Lux.setup(Random.default_rng(), chain[1])
# θinit1, st1 = Lux.setup(Random.default_rng(), chain[2])

# θ1 = [vector_to_parameters(samples[i][1:22], θinit) for i in 50:100]
# θ2 = [vector_to_parameters(samples[i][23:end], θinit1) for i in 50:100]
# tspan = (0.0, 6.0)
# t1 = collect(tspan[1]:0.01:tspan[2])
# luxar1 = [chain[1](t1', θ1[i], st)[1] for i in 1:50]
# luxar2 = [chain[2](t1', θ2[i], st1)[1] for i in 1:50]

# plot(t1, luxar1[500]')
# plot!(t1, luxar2[500]')

# # BPINN 0DE SOLVER COMPARISON CASE
# function lotka_volterra(u, p, t)
#     # Model parameters.
#     α, β, γ, δ = p
#     # Current state.
#     x, y = u

#     # Evaluate differential equations.
#     dx = (α - β * y) * x # prey
#     dy = (δ * x - γ) * y # predator

#     return [dx, dy]
# end

# # initial-value problem.
# u0 = [1.0, 1.0]
# p = [1.5, 1.0, 3.0, 1.0]
# tspan = (0.0, 6.0)
# prob = ODEProblem(lotka_volterra, u0, tspan, p)

# cospit example
@parameters t
@variables u(..)

Dt = Differential(t)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(2 * π * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 4.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1))
initf, re = destructure(chainf)

chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

# non adaptive case
discretization = NeuralPDE.PhysicsInformedNN(chainf, GridTraining([0.01]))
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.01],
    phystd = [0.01],
    priorsNNw = (0.0, 10.0),
    progress = true)

discretization = NeuralPDE.PhysicsInformedNN(chainl, GridTraining([0.01]))
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.01],
    phystd = [0.01],
    priorsNNw = (0.0, 10.0),
    progress = true)

# discretization = NeuralPDE.PhysicsInformedNN(chainf,
#     GridTraining([
#         0.01,
#     ]),
#     adaptive_loss = MiniMaxAdaptiveLoss(2;
#         pde_max_optimiser = Flux.ADAM(1e-4),
#         bc_max_optimiser = Flux.ADAM(0.5),
#         pde_loss_weights = 1,
#         bc_loss_weights = 1,
#         additional_loss_weights = 1)

#     # GradientScaleAdaptiveLoss{Float64, ForwardDiff.Dual{Float64}}(2;
#     #     weight_change_inertia = 0.9,
#     #     pde_loss_weights = ForwardDiff.Dual{Float64}(1.0,
#     #         ntuple(_ -> 0.0, 19)),
#     #     bc_loss_weights = ForwardDiff.Dual{Float64}(1.0,
#     #         ntuple(_ -> 0.0, 19)),
#     #     additional_loss_weights = ForwardDiff.Dual{Float64}(1.0,
#     #         ntuple(_ -> 0.0, 19)))
#     #     GradientScaleAdaptiveLoss{Float64, ForwardDiff.Dual{Float64, Float64}}(2;
#     # weight_change_inertia = 0.9,
#     # pde_loss_weights = ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19)),
#     # bc_loss_weights = ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19)),
#     # additional_loss_weights = ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))))
# )

# # ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))
# # typeof(ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))) <: ForwardDiff.Dual

# a = GradientScaleAdaptiveLoss{Float64, ForwardDiff.Dual{Float64, Float64}}(2;
# weight_change_inertia = 0.9,
# pde_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# zeros(19))),
# bc_loss_weights = ForwardDiff.Dual{Float64}(1.0,
# zeros(19)),
# additional_loss_weights = ForwardDiff.Dual{Float64}(1.0,
#     zeros(19))

# as = ForwardDiff.Dual{Float64}(1.0, ForwardDiff.Partials(ntuple(_ -> 0.0, 19)))
# typeof(ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19)))

# a = GradientScaleAdaptiveLoss{Float64, Float64}(2; weight_change_inertia = 0.9,
#     pde_loss_weights = 1,
#     bc_loss_weights = 1,
#     additional_loss_weights = 1)
# ForwardDiff.Dual{Float64, Float64, 19} <: ForwardDiff.Dual{Float64, Float64}
# typeof(ntuple(_ -> 0.0, 19)) <: Tuple
# ForwardDiff.Dual{Float64}(ForwardDiff.value(1.0), ntuple(_ -> 0.0, 19))
# typeof(ForwardDiff.Dual{Float64}(1.0, ntuple(_ -> 0.0, 19))) <: Real
# cospit example
@parameters t p
@variables u(..)

Dt = Differential(t)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)
eqs = Dt(u(t)) - cos(p * t) ~ 0
bcs = [u(0) ~ 0.0]

domains = [t ∈ Interval(0.0, 4.0)]
chainf = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 1))
initf, re = destructure(chainf)

chainl = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)
initl

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)], [p],
    defaults = Dict(p => 3))

function additional_loss(phi, θ, p)
    # return sum(sum(abs2, phi[i](time', θ[depvars[i]]) .- 1.0) / len for i in 1:1)
    return 2
end
discretization = NeuralPDE.PhysicsInformedNN([chainl],
    GridTraining(0.01),
    # QuadratureTraining(),
    additional_loss = additional_loss,
    param_estim = true)

# discretization.multioutput
pinnrep = NeuralPDE.discretize(pde_system, discretization)

# res = Optimization.solve(pinnrep, BFGS(); callback = callback, maxiters = 5000)
# p_ = res.u[end]
# res.u
# plot!(t1, re(res.u[1:(end - 1)])(t1')')
# depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(pde_system.indvars,
# pde_system.depvars)

ntuple(i -> depvars[i], length(chainl))

[:u]
length(chainl)

ta = range(0.0, 4.0, length = 50)
u = [linear_analytic(0.0, p, ti) for ti in ta]
x̂ = collect(Float64, Array(u) + 0.2 .* Array(u) .* randn(size(u)))
# x̂ = collect(Float64, Array(u) + 0.2  .* randn(size(u)))
time = vec(collect(Float64, ta))
dataset = [x̂, time]
plot!(dataset[2], dataset[1])
# plotly()
# physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

callback = function (p, l)
    println("Current loss is: $l")
    return false
end
res = Optimization.solve(pinnrep, BFGS(); callback = callback, maxiters = 5000)
p_ = res.u[(end - 2):end] # p_ = [9.93, 28.002, 2.667]

mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500, physdt = 1 / 20.0,
    bcstd = [1.0],
    phystd = [0.005], l2std = [0.008], param = [Normal(9, 2)],
    priorsNNw = (0.0, 10.0),
    dataset = dataset,
    progress = true)

typeof((1, 2, 3))
a = [1 2 4 5]'

size(a)
a[1, :]
a[:, 1]
chains = [chainl]
chainn = map(chains) do chain
    Float64.(ComponentArrays.ComponentArray(Lux.initialparameters(Random.default_rng(),
        chain)))
end
names = ntuple(i -> depvars[i], length(chain))
init_params = ComponentArrays.ComponentArray(NamedTuple{names}(i for i in chainn))
init_params isa ComponentVector
mcmc_chain, samples, stats = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.1],
    phystd = [0.01], l2std = [0.01], param = [LogNormal(4, 2)],
    priorsNNw = (0.0, 10.0),
    dataset = dataset,
    progress = true)

tspan = (0.0, 4.0)
t1 = collect(tspan[1]:0.01:tspan[2])

# prior 0-1
# 2000
samples[1000]
# 1500
samples[1000]
# 1000
samples[1500]

# prior 0-10
# 2000
samples[2000]
# 1500
samples[1500]
# 1000
samples[1000]

# prior 0-10
# 2000
samples[2000]
# 1500
samples[1500]
# 1000
samples[1000]

# plot!(t1, chainf(t1')')
# t1
# chainf(t1')'
out1 = re.([samples[i][1:(end - 1)] for i in 1300:1500])
out1 = re.([samples[i][1:(end - 1)] for i in 800:1000])
out1 = re.([samples[i][1:(end)] for i in 800:1000])
luxar1 = collect(out1[i](t1') for i in eachindex(out1))

transsamples = [vector_to_parameters(sample, initl) for sample in samples]
luxar2 = [chainl(t1', transsamples[i], st)[1] for i in 800:1000]

yu = [linear_analytic(0, nothing, t) for t in t1]
plot(t1, yu)
plot!(t1, luxar1[end]')
plot!(t1, luxar2[end]')

using NeuralPDE,
    Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
    Plots
import ModelingToolkit: Interval, infimum, supremum
@parameters t, σ_, β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]
dt = 0.01

input_ = length(domains)
n = 8
chain1 = Flux.Chain(Flux.Dense(input_, n, σ), Flux.Dense(n, n, σ),
    Flux.Dense(n, n, σ),
    Flux.Dense(n, 1))
chain2 = Flux.Chain(Flux.Dense(input_, n, σ), Flux.Dense(n, n, σ),
    Flux.Dense(n, n, σ),
    Flux.Dense(n, 1))
chain3 = Flux.Chain(Flux.Dense(input_, n, σ), Flux.Dense(n, n, σ),
    Flux.Dense(n, n, σ),
    Flux.Dense(n, 1))

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5(), dt = 0.1)
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]

# function getData(sol)
#     data = []
#     us = hcat(sol(ts).u...)
#     ts_ = hcat(sol(ts).t...)
#     return [us, ts_]
# end
# data = getData(sol)

# (u_, t_) = data
# len = length(data[2])

discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3],
    NeuralPDE.GridTraining(dt), param_estim = true)
@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t), z(t)], [σ_, ρ, β],
    defaults = Dict([p .=> 1.0 for p in [σ_, ρ, β]]))
pinnrep = NeuralPDE.discretize(pde_system, discretization)

pinnrep.flat_init_params
pinnrep.loss_functions.pde_loss_functions[1](pinnrep.flat_init_params)
callback = function (p, l)
    println("Current loss is: $l")
    return false
end
res = Optimization.solve(pinnrep, BFGS(); callback = callback, maxiters = 5000)
p_ = res.u[(end - 2):end] # p_ = [9.93, 28.002, 2.667]

using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters t, x, C
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)
eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))

bcs = [u(t, 0) ~ 0.0,# for all t > 0
    u(t, 1) ~ 0.0,# for all t > 0
    u(0, x) ~ x * (1.0 - x), #for all 0 < x < 1
    Dt(u(0, x)) ~ 0.0] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]
@named pde_system = PDESystem(eq,
    bcs,
    domains,
    [t, x],
    [u(t, x)],
    [C],
    defaults = Dict(C => 1.0))

chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ), Lux.Dense(16, 16, Lux.σ), Lux.Dense(16, 1))
discretization = NeuralPDE.PhysicsInformedNN(chain, GridTraining(0.1), param_estim = true)
sym_prob = symbolic_discretize(pde_system, discretization)
sym_prob1 = symbolic_discretize(pde_system, discretization)
println(sym_prob)
println(sym_prob1)

# using NeuralPDE, Lux, ModelingToolkit, DataFrames, CSV, DataLoaders, Flux, IntervalSets,
#     Optimization, OptimizationOptimJL

# # Definisci il modello dell'equazione differenziale
# @parameters t, R, C, Cs

# @variables T_in(..)
# @variables T_ext(..)
# @variables Q_heating(..)
# @variables Q_cooling(..)
# @variables Q_sun(..)
# @variables Q_lights(..)
# @variables Q_equipment(..)

# #details of problem to be solved
# Dt = Differential(t)
# eqs = Dt(T_in(t)) ~ (T_ext(t) - T_in(t)) / (R * C) + Q_heating(t) / C - Q_cooling(t) / C +
#                     Q_sun(t) / Cs + (Q_lights(t) + Q_equipment(t)) / C
# tspan = (0.0, 365.0 * 24.0 * 60.0)  # Dati per un anno
# domains = [t ∈ (0.0, 365.0 * 24.0 * 60.0)]
# bcs = [Dt(T_in(0)) ~ 19.3]

# # dataset creation and additional loss function
# data = CSV.File("shoebox_free.csv") |> DataFrame

# T_ext_data = data."Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"
# Q_heating_data = data."OSGB1000005735772_FLOOR_1_HVAC:Zone Ideal Loads Zone Total Heating Rate [W](TimeStep)"
# Q_cooling_data = data."OSGB1000005735772_FLOOR_1_HVAC:Zone Ideal Loads Zone Total Cooling Rate [W](TimeStep)"
# Q_sun_data = data."Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)"
# Q_lights_data = data."OSGB1000005735772_FLOOR_1:Zone Lights Total Heating Rate [W](TimeStep)"
# Q_equipment_data = data."OSGB1000005735772_FLOOR_1:Zone Electric Equipment Total Heating Rate [W](TimeStep)"

# t_data = 1:size(data, 1)

# dataloader = DataLoader([
#     vcat(T_ext_data,
#         Q_heating_data,
#         Q_cooling_data,
#         Q_sun_data,
#         Q_lights_data,
#         Q_equipment_data,
#         t_data),
# ])

# function additional_loss(phi, θ, p)
#     T_in_predict = phi[1](t_data, θ[1])[1]
#     println(T_in_predict)
#     return sum(abs2(T_in_predict .- T_in_data) / length(data))
# end

# # Creating chain
# input_dim = length(tspan)
# hidden_units = 8
# chain1 = Lux.Chain(Lux.Dense(input_dim, hidden_units, Lux.σ),
#     Lux.Dense(hidden_units, hidden_units, Lux.σ),
#     Lux.Dense(hidden_units, hidden_units, Lux.σ),
#     Lux.Dense(hidden_units, 1))

# # discretize domain with PINN
# dt = 600.0  # 600 secondi (10 minuti)
# discretization = NeuralPDE.PhysicsInformedNN([chain1],
#     NeuralPDE.GridTraining(dt),
#     param_estim = true, additional_loss = additional_loss)

# @named pde_system = NeuralPDE.PDESystem(eqs,
#     bcs,
#     domains, [t],
#     [T_in(t), T_ext(t), Q_heating(t), Q_cooling(t), Q_sun(t), Q_lights(t), Q_equipment(t)],
#     [R, C, Cs],
#     defaults = Dict([R => 1.0, C => 1.0, Cs => 1.0]))

# prob = NeuralPDE.discretize(pde_system, discretization)

# # solve
# res = Optimization.solve(prob,
#     BFGS();
#     maxiters = 5000,
#     callback = (p, l) -> println("Current loss is: $l"))

# # checking solution
# p_optimized = res.u[end]

# minimizer = res.u.depvar[1]
# T_in_predict = minimizer(t_data)

# using Plots
# plot(t_data, T_in_data, label = "Dati Osservati")
# plot!(t_data, T_in_predict, label = "Temperatura Prevista", linestyle = :dash)

# Paper experiments
# function sir_ode!(u, p, t)
#     (S, I, R) = u
#     (β, γ) = p
#     N = S + I + R

#     dS = -β * I / N * S
#     dI = β * I / N * S - γ * I
#     dR = γ * I
#     return [dS, dI, dR]
# end;

# δt = 1.0
# tmax = 40.0
# tspan = (0.0, tmax)
# u0 = [990.0, 10.0, 0.0]; # S,I,R
# p = [0.5, 0.25]; # β,γ (removed c as a parameter as it was just getting multipled with β, so ideal value for c and β taken in new ideal β value)
# prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
# sol = solve(prob_ode, Tsit5(), saveat = δt / 5)
# sig = 0.20
# data = Array(sol)
# dataset = [
#     data[1, :] .+ (minimum(data[1, :]) * sig .* rand(length(sol.t))),
#     data[2, :] .+ (mean(data[2, :]) * sig .* rand(length(sol.t))),
#     data[3, :] .+ (mean(data[3, :]) * sig .* rand(length(sol.t))),
#     sol.t,
# ]
# priors = [Normal(1.0, 1.0), Normal(0.5, 1.0)]

# plot(sol.t, dataset[1], label = "noisy s")
# plot!(sol.t, dataset[2], label = "noisy i")
# plot!(sol.t, dataset[3], label = "noisy r")
# plot!(sol, labels = ["s" "i" "r"])

# chain = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 8, tanh),
#     Flux.Dense(8, 3))

# Adaptorkwargs = (Adaptor = AdvancedHMC.StanHMCAdaptor,
#     Metric = AdvancedHMC.DiagEuclideanMetric, targetacceptancerate = 0.8)

# alg = BNNODE(chain;
#     dataset = dataset,
#     draw_samples = 500,
#     l2std = [5.0, 5.0, 10.0],
#     phystd = [1.0, 1.0, 1.0],
#     priorsNNw = (0.01, 3.0),
#     Adaptorkwargs = Adaptorkwargs,
#     param = priors, progress = true)

# # our version
# @time sol_pestim3 = solve(prob_ode, alg; estim_collocate = true, saveat = δt)
# @show sol_pestim3.estimated_ode_params

# # old version
# @time sol_pestim4 = solve(prob_ode, alg; saveat = δt)
# @show sol_pestim4.estimated_ode_params

# # plotting solutions
# plot(sol_pestim3.ensemblesol[1], label = "estimated x1")
# plot!(sol_pestim3.ensemblesol[2], label = "estimated y1")
# plot!(sol_pestim3.ensemblesol[3], label = "estimated z1")

# plot(sol_pestim4.ensemblesol[1], label = "estimated x2_1")
# plot!(sol_pestim4.ensemblesol[2], label = "estimated y2_1")
# plot!(sol_pestim4.ensemblesol[3], label = "estimated z2_1")

# using NeuralPDE, Lux, ModelingToolkit, DataFrames, CSV, DataLoaders, Flux, IntervalSets,
#     Optimization, OptimizationOptimJL

# # Definisci il modello dell'equazione differenziale
# @parameters t, R, C, Cs

# @variables T_in(..),
# T_ext(..),
# Q_heating(..),
# Q_cooling(..),
# Q_sun(..),
# Q_lights(..),
# Q_equipment(..)

# # R, C, Cs = [1, 2, 3]

# Dt = Differential(t)

# # eqs = Dt(T_in(t)) ~ (-T_ext(t) + T_in(t)) / (R * C)
# eqs = Dt(T_in(t)) ~ (T_ext(t) - T_in(t)) / (R * C) + Q_heating(t) / C - Q_cooling(t) / C +
#                     Q_sun(t) / Cs + (Q_lights(t) + Q_equipment(t)) / C

# domains = [t ∈ Interval(0.0, 365.0 * 24.0 * 60.0)]
# bcs = [Dt(T_in(0.0)) ~ 4.48]

# dt = 10.0  # 600 seconds (10 minute)

# # Define the temporal space
# tspan = (0.0, 365.0 * 24.0 * 60.0)  # Dati per un anno

# # load sampled data from CSV
# data = CSV.File("shoebox_free.csv") |> DataFrame

# # Put the sampled data in dedicated variables
# T_in_data = data."OSGB1000005735772_FLOOR_1:Zone Mean Air Temperature [C](TimeStep)"
# T_ext_data = data."Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"
# Q_heating_data = data."OSGB1000005735772_FLOOR_1_HVAC:Zone Ideal Loads Zone Total Heating Rate [W](TimeStep)"
# Q_cooling_data = data."OSGB1000005735772_FLOOR_1_HVAC:Zone Ideal Loads Zone Total Cooling Rate [W](TimeStep)"
# Q_sun_data = data."Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)"
# Q_lights_data = data."OSGB1000005735772_FLOOR_1:Zone Lights Total Heating Rate [W](TimeStep)"
# Q_equipment_data = data."OSGB1000005735772_FLOOR_1:Zone Electric Equipment Total Heating Rate [W](TimeStep)"

# t_data = collect(Float64, 1:size(data, 1))

# dataloader = DataLoader([
#     vcat(T_in_data,
#         T_ext_data,
#         Q_heating_data,
#         Q_cooling_data,
#         Q_sun_data,
#         Q_lights_data,
#         Q_equipment_data,
#         t_data),
# ])

# dataloader
# # Define the NN
# input_dim = 1
# hidden_units = 8
# len = length(t_data)

# chain1 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64
# chain2 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64
# chain3 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64
# chain4 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64
# chain5 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64
# chain6 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64
# chain7 = Flux.Chain(Flux.Dense(input_dim, hidden_units, σ), Flux.Dense(hidden_units, 1)) |>
#          Flux.f64

# #Define dependent and independent vatiables
# indvars = [t]
# depvars = [:T_in, :T_ext, :Q_heating, :Q_cooling, :Q_sun, :Q_lights, :Q_equipment]
# u_ = hcat(T_in_data,
#     T_ext_data,
#     Q_heating_data,
#     Q_cooling_data,
#     Q_sun_data,
#     Q_lights_data,
#     Q_equipment_data)

# # Define the loss(additional loss will be using all data vectors)
# init_params = [Flux.destructure(c)[1]
#                for c in [chain1, chain2, chain3, chain4, chain5, chain6, chain7]]
# acum = [0; accumulate(+, length.(init_params))]
# sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]

# function additional_loss(phi, θ, p)
#     return sum(sum(abs2, phi[i](t_data[1:500]', θ[sep[i]]) .- u_[:, [i]]') / 500.0
#                for i in 1:1:1)
# end

# @named pde_system = NeuralPDE.PDESystem(eqs,
#     bcs,
#     domains,
#     [t],
#     [T_in(t), T_ext(t), Q_heating(t), Q_cooling(t), Q_sun(t), Q_lights(t), Q_equipment(t)
#     ],
#     [R, C, Cs],
#     defaults = Dict([R => 1.0, C => 1.0, Cs => 1.0]))#[R, C, Cs])

# discretization = NeuralPDE.PhysicsInformedNN([
#         chain1,
#         chain2,
#         chain3,
#         chain4,
#         chain5,
#         chain6, chain7,
#     ],
#     NeuralPDE.GridTraining(dt), param_estim = true, additional_loss = additional_loss)
# prob = NeuralPDE.discretize(pde_system, discretization)

# # Parameter Optimization
# res = Optimization.solve(prob,
#     BFGS(),
#     maxiters = 1000,
#     callback = callback)

# p_optimized = res.u[end]
# # Plot fo results
# minimizer = res.u.depvar[1]
# T_in_predict = minimizer(t_data)

# using Plots
# plot(t_data, T_in_data, label = "Dati Osservati")
# plot!(t_data, T_in_predict, label = "Temperatura Prevista", linestyle = :dash)
