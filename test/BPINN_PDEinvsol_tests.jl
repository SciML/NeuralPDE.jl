using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays, ModelingToolkit

Random.seed!(100)

# Cos(pit) periodic curve (Parameter Estimation)
println("Example 1, 2d Periodic System")
@parameters t, p
@variables u(..)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(p * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1)) |> Flux.f64
init1, re1 = Flux.destructure(chainf)
chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs,
    bcs,
    domains,
    [t],
    [u(t)],
    [p],
    defaults = Dict([p => 4.0]))

analytic_sol_func1(u0, t) = u0 + sin(2 * π * t) / (2 * π)
timepoints = collect(0.0:(1 / 100.0):2.0)
u1 = [analytic_sol_func1(0.0, timepoint) for timepoint in timepoints]
u1 = u1 .+ (u1 .* 0.2) .* randn(size(u1))
dataset = [hcat(u1, timepoints)]

# plot(dataset[1][:, 2], dataset[1][:, 1])
# plot!(timepoints, u)

# checking all training strategies
discretization = NeuralPDE.BayesianPINN([chainl],
    StochasticTraining(200),
    param_estim = true, dataset = [dataset, nothing])

ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)], progress = true)

discretization = NeuralPDE.BayesianPINN([chainl],
    QuasiRandomTraining(200),
    param_estim = true, dataset = [dataset, nothing])

ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainl],
    QuadratureTraining(), param_estim = true, dataset = [dataset, nothing])

ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainl],
    GridTraining([0.02]),
    param_estim = true, dataset = [dataset, nothing])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

discretization = NeuralPDE.BayesianPINN([chainf],
    GridTraining([0.02]), param_estim = true, dataset = [dataset, nothing])

sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.03],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)])

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])

@test u_predict≈u_real atol=1.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3

ts = vec(sol2.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol2.ensemblesol[1])

@test u_predict≈u_real atol=0.5
@test mean(u_predict .- u_real) < 0.1
@test sol2.estimated_de_params[1]≈param atol=param * 0.3

## Example Lorenz System (Parameter Estimation)
println("Example 2, Lorenz System")
@parameters t, σ_
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (28.0 - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - 8 / 3 * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]

input_ = length(domains)
n = 7
chain = [
    Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.tanh),
        Lux.Dense(n, 1)),
    Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.tanh),
        Lux.Dense(n, 1)),
    Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.tanh),
        Lux.Dense(n, 1)),
]

#Generate Data
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5(), dt = 0.01, saveat = 0.05)
ts = sol.t
us = hcat(sol.u...)
us = us .+ ((0.05 .* randn(size(us))) .* us)
ts_ = hcat(sol(ts).t...)[1, :]
dataset = [hcat(us[i, :], ts_) for i in 1:3]

# using Plots, StatsPlots
# plot(hcat(sol.u...)[1, :], hcat(sol.u...)[2, :], hcat(sol.u...)[3, :])
# plot!(dataset[1][:, 1], dataset[2][:, 1], dataset[3][:, 1])
# plot(dataset[1][:, 2:end], dataset[1][:, 1])
# plot!(dataset[2][:, 2:end], dataset[2][:, 1])
# plot!(dataset[3][:, 2:end], dataset[3][:, 1])

discretization = NeuralPDE.BayesianPINN(chain, NeuralPDE.GridTraining([0.01]);
    param_estim = true, dataset = [dataset, nothing])

@named pde_system = PDESystem(eqs, bcs, domains,
    [t], [x(t), y(t), z(t)], [σ_], defaults = Dict([p => 1.0 for p in [σ_]]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 20,
    bcstd = [0.3, 0.3, 0.3],
    phystd = [0.1, 0.1, 0.1],
    l2std = [1, 1, 1],
    priorsNNw = (0.0, 1.0),
    saveats = [0.01],
    param = [Normal(12.0, 2)])

idealp = 10.0
p_ = sol1.estimated_de_params[1]

# plot(pmean(sol1.ensemblesol[1]), pmean(sol1.ensemblesol[2]), pmean(sol1.ensemblesol[3]))
# plot(sol1.timepoints[1]', pmean(sol1.ensemblesol[1]))
# plot!(sol1.timepoints[2]', pmean(sol1.ensemblesol[2]))
# plot!(sol1.timepoints[3]', pmean(sol1.ensemblesol[3]))

@test sum(abs, pmean(p_) - 10.00) < 0.3 * idealp[1]
# @test sum(abs, pmean(p_[2]) - (8 / 3)) < 0.3 * idealp[2]

# # NEW LOSS FUNCTION CODE
# pinnrep = symbolic_discretize(pde_system, discretization)

# # general equation with diff
# # now 1> substute u(t), phi(t) values from dataset and get multiple equations
# # phi[i] must be in numeric_derivative() form
# # derivative(phi, u, [x, y], εs, order, θ) - use parse_equations() and interp object to create loss function
# # this function must take interp objects(train sets)
# # dataset - get u(t), t from dataset interpolations object
# # make lhs-rhs loss
# # sum losses

# using DataInterpolations

# # dataset_pde has normal matrix format 
# # dataset_bc has format of Vector{typeof(dataset_pde )} as each bc has different domain requirements
# function get_symbols(dict_depvar_input, dataset, depvars)
#     # get datasets into splattable form
#     splat_form = [[dataset_i[:, i] for i in 1:size(dataset_i)[2]] for dataset_i in dataset]
#     # splat datasets onto Linear interpolations tables
#     interps = [LinearInterpolation(splat_i...) for splat_i in splat_form]
#     interps = Dict(depvars .=> interps)

#     Dict_symbol_interps = Dict(depvar => (interps[depvar], dict_depvar_input[depvar])
#                                for depvar in depvars)

#     tobe_subs = Dict()
#     for (a, b) in dict_depvar_input
#         tobe_subs[a] = eval(:($a($(b...))))
#     end

#     to_subs = Dict()
#     for (a, b) in Dict_symbol_interps
#         b1, b2 = b
#         to_subs[a] = eval(:($b1($(b2...))))
#     end
#     return to_subs, tobe_subs
# end

# function recur_expression(exp, Dict_differentials)
#     for in_exp in exp.args
#         if !(in_exp isa Expr)
#             # skip +,== symbols, characters etc
#             continue

#         elseif in_exp.args[1] isa ModelingToolkit.Differential
#             # first symbol of differential term
#             # Dict_differentials for masking differential terms
#             # and resubstituting differentials in equations after putting in interpolations
#             Dict_differentials[eval(in_exp)] = Symbol("diff_$(length(Dict_differentials)+1)")
#             return

#         else
#             recur_expression(in_exp, Dict_differentials)
#         end
#     end
# end

# # get datafree loss functions for new loss type
# # need to call merge_strategy_with_loss_function() variant after this
# function merge_dataset_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
#         dataset,
#         datafree_pde_loss_function,
#         datafree_bc_loss_function)
#     @unpack domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params = pinnrep

#     eltypeθ = eltype(pinnrep.flat_init_params)

#     train_sets = [[dataset[i][:, 2] for i in eachindex(dataset)], [[0;;], [0;;], [0;;]]]

#     # the points in the domain and on the boundary
#     pde_train_sets, bcs_train_sets = train_sets
#     # pde_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
#     #     pde_train_sets)
#     # bcs_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
#     #     bcs_train_sets)
#     pde_loss_functions = [get_loss_function(_loss, _set, eltypeθ)
#                           for (_loss, _set) in zip(datafree_pde_loss_function,
#         pde_train_sets)]

#     bc_loss_functions = [get_loss_function(_loss, _set, eltypeθ)
#                          for (_loss, _set) in zip(datafree_bc_loss_function, bcs_train_sets)]

#     pde_loss_functions, bc_loss_functions
# end

# function get_loss_function(loss_function, train_set, eltypeθ; τ = nothing)
#     loss = (θ) -> mean(abs2, loss_function(train_set, θ))
# end

# # for bc case, [bc]/bc eqs must be passed along with dataset_bc[i]
# # and final loss for bc must be together in a vector(bcs has seperate type of dataset_bc)
# # eqs is vector of pde eqs and dataset here is dataset_pde
# # normally you get vector of losses
# function get_loss_2(pinnrep, dataset, eqs)
#     depvars = pinnrep.depvars # order is same as dataset and interps
#     dict_depvar_input = pinnrep.dict_depvar_input

#     to_subs, tobe_subs = get_symbols(dict_depvar_input, dataset, depvars)
#     interp_subs_dict = Dict(tobe_subs[depvar] => to_subs[depvar] for depvar in depvars)

#     Dict_differentials = Dict()
#     exp = toexpr(eqs)
#     void_value = [recur_expression(exp_i, Dict_differentials) for exp_i in exp]
#     # Dict_differentials is now filled with Differential operator => diff_i key-value pairs

#     # masking operation
#     a = substitute.(eqs, Ref(Dict_differentials))
#     b = substitute.(a, Ref(interp_subs_dict))
#     # reverse dict for re-substituing values of Differential(t)(u(t)) etc
#     rev_Dict_differentials = Dict(value => key for (key, value) in Dict_differentials)
#     eqs = substitute.(b, Ref(rev_Dict_differentials))
#     # get losses
#     loss_functions = [NeuralPDE.build_loss_function(pinnrep,
#         eqs[i],
#         pinnrep.pde_indvars[i]) for i in eachindex(eqs)]
# end

# eqs = pde_system.eqs
# yuh1 = get_loss_2(pinnrep, dataset, eqs)
# eqs = pinnrep.bcs
# yuh2 = get_loss_2(pinnrep, dataset, eqs)

# pde_loss_functions, bc_loss_functions = merge_dataset_with_loss_function(pinnrep,
#     dataset,
#     yuh1,
#     yuh2)

# pde_loss_functions()
# # logic for recursion formula to parse differentials
# # # this below has the whole differential term
# # toexpr(pde_system.eqs[1]).args[2].args[3].args[3] isa ModelingToolkit.Differential
# # toexpr(pde_system.eqs[1]).args[2].args[3].args[3]
# # # .args[1] isa ModelingToolkit.Differential

# # logic for interpolation and indvars splatting to get Equation parsing terms
# # splat_form = [[dataset_i[:, i] for i in 1:size(dataset_i)[2]] for dataset_i in dataset]
# # # splat datasets onto Linear interpolations tables
# # interps = [LinearInterpolation(splat_i...) for splat_i in splat_form]
# # interps = Dict(depvars .=> interps)
# # get datasets into splattable form
# # splat_form = [[dataset_i[:, i] for i in 1:size(dataset_i)[2]] for dataset_i in dataset]
# # # splat datasets onto Linear interpolations tables
# # yu = [LinearInterpolation(splat_i...) for splat_i in splat_form]
# # Symbol(:($(yu[1])))

# # logic to contrauct dict to feed for masking
# # Dict(interps[depvar] => dict_depvar_input[depvar] for depvar in depvars)