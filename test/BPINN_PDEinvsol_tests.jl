using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays, ModelingToolkit

Random.seed!(100)

@testset "Example 1: 2D Periodic System with parameter estimation" begin
    # Cos(pi*t) periodic curve
    @parameters t, p
    @variables u(..)

    Dt = Differential(t)
    eqs = Dt(u(t)) - cos(p * t) ~ 0
    bcs = [u(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

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

    # checking all training strategies
    discretization = BayesianPINN([chainl], StochasticTraining(200), param_estim = true,
        dataset = [dataset, nothing])

    ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 1500,
        bcstd = [0.05],
        phystd = [0.01], l2std = [0.01],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 50.0],
        param = [LogNormal(6.0, 0.5)])

    discretization = BayesianPINN([chainl], QuasiRandomTraining(200), param_estim = true,
        dataset = [dataset, nothing])

    ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 1500,
        bcstd = [0.05],
        phystd = [0.01], l2std = [0.01],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 50.0],
        param = [LogNormal(6.0, 0.5)])

    discretization = BayesianPINN([chainl], QuadratureTraining(), param_estim = true,
        dataset = [dataset, nothing])

    ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 1500,
        bcstd = [0.05],
        phystd = [0.01], l2std = [0.01],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 50.0],
        param = [LogNormal(6.0, 0.5)])

    discretization = BayesianPINN([chainl], GridTraining([0.02]), param_estim = true,
        dataset = [dataset, nothing])

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 1500,
        bcstd = [0.05],
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
end

@testset "Example 2: Lorenz System with parameter estimation" begin
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

    discretization = BayesianPINN(chain, GridTraining([0.01]); param_estim = true,
        dataset = [dataset, nothing])

    @named pde_system = PDESystem(eqs, bcs, domains,
        [t], [x(t), y(t), z(t)], [σ_], defaults = Dict([p => 1.0 for p in [σ_]]))

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 50,
        bcstd = [0.3, 0.3, 0.3],
        phystd = [0.1, 0.1, 0.1],
        l2std = [1, 1, 1],
        priorsNNw = (0.0, 1.0),
        saveats = [0.01],
        param = [Normal(12.0, 2)])

    idealp = 10.0
    p_ = sol1.estimated_de_params[1]
    @test sum(abs, pmean(p_) - 10.00) < 0.3 * idealp[1]
    # @test sum(abs, pmean(p_[2]) - (8 / 3)) < 0.3 * idealp[2]
end

@parameters t, x, p
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
eqs = [u(t, x) * Dt(u(t, x)) - cos(p * t) ~ 0, u(t, x) + Dx(u(t, x)) ~ 0.0]
bcs = [u(0, x) ~ 0.0, u(t, 10) ~ 1.0]
domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 2.0)]

chainl = Lux.Chain(Lux.Dense(2, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs,
    bcs,
    domains,
    [t, x],
    [u(t, x)],
    [p],
    defaults = Dict([p => 4.0]))

analytic_sol_func1(u0, t) = u0 + sin(2 * π * t) / (2 * π)
timepoints = collect(0.0:(1 / 100.0):2.0)
u1 = [analytic_sol_func1(0.0, timepoint) for timepoint in timepoints]
u1 = u1 .+ (u1 .* 0.2) .* randn(size(u1))
dataset = [hcat(u1, u1, timepoints)]

# checking all training strategies
# discretization = BayesianPINN([chainl], GridTraining([0.02]), param_estim = true,
#     dataset = [dataset, nothing])

discretization = BayesianPINN([chainl],
    GridTraining([0.2, 0.2]),
    param_estim = true, dataset = [dataset, nothing])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05, 0.05],
    phystd = [0.01, 0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0, 1 / 20.0],
    param = [Normal(3.0, 0.5)], progress = true)

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])

@test u_predict≈u_real atol=1.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3

# for bc case, [bc]/bc eqs must be passed along with dataset_bc[i]
# and final loss for bc must be together in a vector(bcs has seperate type of dataset_bc)
# eqs is vector of pde eqs and dataset here is dataset_pde
# normally you get vector of losses
function get_loss_2(pinnrep, dataset, eqs)
    depvars = pinnrep.depvars # order is same as dataset and interps
    dict_depvar_input = pinnrep.dict_depvar_input

    to_subs, tobe_subs = get_symbols(dict_depvar_input, dataset, depvars)
    interp_subs_dict = Dict(tobe_subs[depvar] => to_subs[depvar] for depvar in depvars)

    Dict_differentials = Dict()
    exp = toexpr(eqs)
    void_value = [recur_expression(exp_i, Dict_differentials) for exp_i in exp]
    # Dict_differentials is now filled with Differential operator => diff_i key-value pairs

    # masking operation
    a = substitute.(eqs, Ref(Dict_differentials))
    println(a)
    b = substitute.(a, Ref(interp_subs_dict))
    println(b)
    # reverse dict for re-substituing values of Differential(t)(u(t)) etc
    rev_Dict_differentials = Dict(value => key for (key, value) in Dict_differentials)
    eqs = substitute.(b, Ref(rev_Dict_differentials))
    # get losses
    loss_functions = [NeuralPDE.build_loss_function(pinnrep,
        eqs[i],
        pinnrep.pde_indvars[i]) for i in eachindex(eqs)]
end

# >why not mask differential
function get_lossy(pinnrep, dataset, eqs)
    depvars = pinnrep.depvars # order is same as dataset and interps
    dict_depvar_input = pinnrep.dict_depvar_input

    Dict_differentials = Dict()
    exp = toexpr(eqs)
    for exp_i in exp
        recur_expression(exp_i, Dict_differentials)
    end
    # Dict_differentials is now filled with Differential operator => diff_i key-value pairs

    # masking operation
    println("Dict_differentials : ", Dict_differentials)
    a = substitute.(eqs, Ref(Dict_differentials))
    println("Masked Differential term : ", a)

    to_subs, tobe_subs = get_symbols(dict_depvar_input, dataset, depvars, eqs)
    # for each row in dataset create u values for substituing in equation, n_equations=n_rows
    eq_subs = [Dict(tobe_subs[depvar] => to_subs[depvar][i] for depvar in depvars)
               for i in 1:size(dataset[1][:, 1])[1]]

    b = []
    for eq_sub in eq_subs
        push!(b, [substitute(a_i, eq_sub) for a_i in a])
    end

    # reverse dict for re-substituing values of Differential(t)(u(t)) etc
    rev_Dict_differentials = Dict(value => key for (key, value) in Dict_differentials)

    c = []
    for b_i in b
        push!(c, substitute.(b_i, Ref(rev_Dict_differentials)))
    end
    println("After re Substituting depvars : ", c[1])
    # c = vcat(c...)
    println(c)
    c
    # get losses
    # loss_functions = [NeuralPDE.build_loss_function(pinnrep,
    #     c[i, :][j],
    #     pinnrep.pde_indvars[j]) for j in eachindex(pinnrep.pde_indvars)]
    # return loss_functions
end

# dataset_pde has normal matrix format 
# dataset_bc has format of Vector{typeof(dataset_pde )} as each bc has different domain requirements
function get_symbols(dict_depvar_input, dataset, depvars, eqs)
    depvar_vals = [dataset_i[:, 1] for dataset_i in dataset]
    # order of depvars
    to_subs = Dict(pinnrep.depvars .=> depvar_vals)

    asrt = Symbolics.get_variables.(eqs)
    # want only symbols of depvars
    temp = unique(reduce(vcat, asrt))
    # now we have all the depvars, we now need all depvars whcih can be substituted with data interps

    tobe_subs = Dict()
    for a in depvars
        for i in temp
            expr = toexpr(i)
            if (expr isa Expr) && (expr.args[1] == a)
                tobe_subs[a] = i
            end
        end
    end

    return to_subs, tobe_subs
end

yuh = get_symbols(pinnrep.dict_depvar_input, dataset, pinnrep.depvars, pinnrep.eqs)

function recur_expression(exp, Dict_differentials)
    for in_exp in exp.args
        if !(in_exp isa Expr)
            # skip +,== symbols, characters etc
            continue

        elseif in_exp.args[1] isa ModelingToolkit.Differential
            # first symbol of differential term
            # Dict_differentials for masking differential terms
            # and resubstituting differentials in equations after putting in interpolations
            # temp = in_exp.args[end]
            # in_exp.args[end] = Symbolics.variable(in_exp.args[end])

            Dict_differentials[in_exp] = Symbolics.variable("diff_$(length(Dict_differentials)+1)")
            return
        else
            recur_expression(in_exp, Dict_differentials)
        end
    end
end
vars = Symbolics.variable.(hcat(pinnrep.indvars, pinnrep.depvars))
toexpr(Differential(t)(Differential(u)(u(t))) + u(t) ~ 0).args[2]
eqs
# Differential(t)(u(t)) - cos(p * t) ~ 0
exprs = toexpr(eqs)
pop = Dict()
recur_expression(exprs, pop)
pop1 = Dict()
for (a, b) in pop
    pop1[eval(a)] = b
end
pop1
a = substitute(eqs, pop1)

Symbolics.get_variables(eqs[1])
# eqs=a

NeuralPDE.get_variables(pinnrep.eqs, pinnrep.dict_indvars, pinnrep.dict_depvars)
NeuralPDE.get_argument(pinnrep.bcs, pinnrep.dict_indvars, pinnrep.dict_depvars)
dx = pinnrep.strategy.dx
eltypeθ = eltype(pinnrep.flat_init_params)

# solve dataset physics loss for heterogenous case
# create number of equations as number of interpolation and points(n rows)
# follow masking and finally feed training sets as set in interpolations input of u(t,x,..)

# logic for recursion formula to parse differentials
# # this below has the whole differential term
toexpr(pde_system.eqs[1]).args[2].args[3].args[3]
# toexpr(pde_system.eqs[1]).args[2].args[3].args[3]
# # .args[1] isa ModelingToolkit.Differential

# logic for interpolation and indvars splatting to get Equation parsing terms
# splat_form = [[dataset_i[:, i] for i in 1:size(dataset_i)[2]] for dataset_i in dataset]
# # splat datasets onto Linear interpolations tables
# interps = [LinearInterpolation(splat_i...) for splat_i in splat_form]
# interps = Dict(depvars .=> interps)
# get datasets into splattable form
# splat_form = [[dataset_i[:, i] for i in 1:size(dataset_i)[2]] for dataset_i in dataset]
# # splat datasets onto Linear interpolations tables
# yu = [LinearInterpolation(splat_i...) for splat_i in splat_form]
# Symbol(:($(yu[1])))

# logic to contrauct dict to feed for masking
# Dict(interps[depvar] => dict_depvar_input[depvar] for depvar in depvars)

@parameters t, p
@variables u(..)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(p * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]

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

# checking all training strategies
discretization = BayesianPINN([chainl], GridTraining(0.01), param_estim = true,
    dataset = [dataset, nothing])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)], progress = true)