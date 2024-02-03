using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays
using Flux

Random.seed!(100)

@testset "Example 1: 2D Periodic System" begin
    # Cos(pi*t) example
    @parameters t
    @variables u(..)
    Dt = Differential(t)
    eqs = Dt(u(t)) - cos(2 * π * t) ~ 0
    bcs = [u(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]
    chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
    initl, st = Lux.setup(Random.default_rng(), chainl)
    @named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

    # non adaptive case
    discretization = BayesianPINN([chainl], GridTraining([0.01]))

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 1500,
        bcstd = [0.02],
        phystd = [0.01],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 50.0])

    analytic_sol_func(u0, t) = u0 + sin(2 * π * t) / (2 * π)
    ts = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(0.0, t) for t in ts]
    u_predict = pmean(sol1.ensemblesol[1])
    @test u_predict≈u_real atol=0.5
    @test mean(u_predict .- u_real) < 0.1
end

@testset "Example 2: 1D ODE" begin
    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~ θ^3 + 2 * θ + (θ^2) * ((1 + 3 * (θ^2)) / (1 + θ + (θ^3))) -
                    u(θ) * (θ + ((1 + 3 * (θ^2)) / (1 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = Lux.Chain(Lux.Dense(1, 12, Lux.σ), Lux.Dense(12, 1))

    discretization = BayesianPINN([chain], GridTraining([0.01]))

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 500,
        bcstd = [0.1],
        phystd = [0.05],
        priorsNNw = (0.0, 10.0),
        saveats = [1 / 100.0])

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol1.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol1.ensemblesol[1])
    @test u_predict≈u_real atol=0.8
end

@testset "Example 3: 3rd Degree ODE" begin
    @parameters x
    @variables u(..), Dxu(..), Dxxu(..), O1(..), O2(..)
    Dxxx = Differential(x)^3
    Dx = Differential(x)

    # ODE
    eq = Dx(Dxxu(x)) ~ cos(pi * x)

    # Initial and boundary conditions
    ep = (cbrt(eps(eltype(Float64))))^2 / 6

    bcs = [u(0.0) ~ 0.0,
        u(1.0) ~ cos(pi),
        Dxu(1.0) ~ 1.0,
        Dxu(x) ~ Dx(u(x)) + ep * O1(x),
        Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x)]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = [
        Lux.Chain(Lux.Dense(1, 10, Lux.tanh), Lux.Dense(10, 10, Lux.tanh),
            Lux.Dense(10, 1)), Lux.Chain(Lux.Dense(1, 10, Lux.tanh), Lux.Dense(10, 10, Lux.tanh),
            Lux.Dense(10, 1)), Lux.Chain(Lux.Dense(1, 10, Lux.tanh), Lux.Dense(10, 10, Lux.tanh),
            Lux.Dense(10, 1)), Lux.Chain(Lux.Dense(1, 4, Lux.tanh), Lux.Dense(4, 1)),
        Lux.Chain(Lux.Dense(1, 4, Lux.tanh), Lux.Dense(4, 1))]

    discretization = BayesianPINN(chain, GridTraining(0.01))

    @named pde_system = PDESystem(eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)])

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 200,
        bcstd = [0.01, 0.01, 0.01, 0.01, 0.01],
        phystd = [0.005],
        priorsNNw = (0.0, 10.0),
        saveats = [1 / 100.0])

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

    u_predict = pmean(sol1.ensemblesol[1])
    xs = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(x) for x in xs]
    @test u_predict≈u_real atol=0.5
end

@testset "Example 4: 2D Poissons equation" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    # Boundary conditions
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    # Neural network
    dim = 2 # number of dimensions
    chain = Lux.Chain(Lux.Dense(dim, 9, Lux.σ), Lux.Dense(9, 9, Lux.σ), Lux.Dense(9, 1))

    # Discretization
    dx = 0.04
    discretization = BayesianPINN([chain], GridTraining(dx))

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 200,
        bcstd = [0.003, 0.003, 0.003, 0.003],
        phystd = [0.003],
        priorsNNw = (0.0, 10.0),
        saveats = [1 / 100.0, 1 / 100.0])

    xs = sol1.timepoints[1]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

    u_predict = pmean(sol1.ensemblesol[1])
    u_real = [analytic_sol_func(xs[:, i][1], xs[:, i][2]) for i in 1:length(xs[1, :])]
    @test u_predict≈u_real atol=1.5
end

@testset "Translating from Flux" begin
    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~ θ^3 + 2 * θ + (θ^2) * ((1 + 3 * (θ^2)) / (1 + θ + (θ^3))) -
                    u(θ) * (θ + ((1 + 3 * (θ^2)) / (1 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = Flux.Chain(Flux.Dense(1, 12, Flux.σ), Flux.Dense(12, 1))

    discretization = BayesianPINN([chain], GridTraining([0.01]))
    @test discretization.chain[1] isa Lux.AbstractExplicitLayer

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol1 = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 500,
        bcstd = [0.1],
        phystd = [0.05],
        priorsNNw = (0.0, 10.0),
        saveats = [1 / 100.0])

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol1.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol1.ensemblesol[1])
    @test u_predict≈u_real atol=0.8
end



# # NEW LOSS FUNCTION CODE
pinnrep = symbolic_discretize(pde_system, discretization)

# general equation with diff
# now 1> substute u(t), phi(t) values from dataset and get multiple equations
# phi[i] must be in numeric_derivative() form
# derivative(phi, u, [x, y], εs, order, θ) - use parse_equations() and interp object to create loss function
# this function must take interp objects(train sets)
# dataset - get u(t), t from dataset interpolations object
# make lhs-rhs loss
# sum losses

using DataInterpolations

# dataset_pde has normal matrix format 
# dataset_bc has format of Vector{typeof(dataset_pde )} as each bc has different domain requirements
function get_symbols(dict_depvar_input, dataset, depvars)
    # get datasets into splattable form
    splat_form = [[dataset_i[:, i] for i in 1:size(dataset_i)[2]] for dataset_i in dataset]
    # splat datasets onto Linear interpolations tables
    interps = [LinearInterpolation(splat_i...) for splat_i in splat_form]
    interps = Dict(depvars .=> interps)

    Dict_symbol_interps = Dict(depvar => (interps[depvar], dict_depvar_input[depvar])
                               for depvar in depvars)

    tobe_subs = Dict()
    for (a, b) in dict_depvar_input
        tobe_subs[a] = eval(:($a($(b...))))
    end

    to_subs = Dict()
    for (a, b) in Dict_symbol_interps
        b1, b2 = b
        to_subs[a] = eval(:($b1($(b2...))))
    end
    return to_subs, tobe_subs
end

function recur_expression(exp, Dict_differentials)
    for in_exp in exp.args
        if !(in_exp isa Expr)
            # skip +,== symbols, characters etc
            continue

        elseif in_exp.args[1] isa ModelingToolkit.Differential
            # first symbol of differential term
            # Dict_differentials for masking differential terms
            # and resubstituting differentials in equations after putting in interpolations
            Dict_differentials[eval(in_exp)] = Symbol("diff_$(length(Dict_differentials)+1)")
            return

        else
            recur_expression(in_exp, Dict_differentials)
        end
    end
end

# get datafree loss functions for new loss type
# need to call merge_strategy_with_loss_function() variant after this
function merge_dataset_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
        dataset,
        datafree_pde_loss_function,
        datafree_bc_loss_function)
    @unpack domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    train_sets = [[dataset[i][:, 2] for i in eachindex(dataset)], [[0;;], [0;;], [0;;]]]

    # the points in the domain and on the boundary
    pde_train_sets, bcs_train_sets = train_sets
    # pde_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
    #     pde_train_sets)
    # bcs_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
    #     bcs_train_sets)
    pde_loss_functions = [get_loss_function(_loss, _set, eltypeθ)
                          for (_loss, _set) in zip(datafree_pde_loss_function,
        pde_train_sets)]

    bc_loss_functions = [get_loss_function(_loss, _set, eltypeθ)
                         for (_loss, _set) in zip(datafree_bc_loss_function, bcs_train_sets)]

    pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, train_set, eltypeθ; τ = nothing)
    loss = (θ) -> mean(abs2, loss_function(train_set, θ))
end

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
    b = substitute.(a, Ref(interp_subs_dict))
    # reverse dict for re-substituing values of Differential(t)(u(t)) etc
    rev_Dict_differentials = Dict(value => key for (key, value) in Dict_differentials)
    eqs = substitute.(b, Ref(rev_Dict_differentials))
    # get losses
    loss_functions = [NeuralPDE.build_loss_function(pinnrep,
        eqs[i],
        pinnrep.pde_indvars[i]) for i in eachindex(eqs)]
end

eqs = pde_system.eqs
yuh1 = get_loss_2(pinnrep, dataset, eqs)
eqs = pinnrep.bcs
yuh2 = get_loss_2(pinnrep, dataset, eqs)

pde_loss_functions, bc_loss_functions = merge_dataset_with_loss_function(pinnrep,
    dataset,
    yuh1,
    yuh2)

pde_loss_functions()
# logic for recursion formula to parse differentials
# # this below has the whole differential term
# toexpr(pde_system.eqs[1]).args[2].args[3].args[3] isa ModelingToolkit.Differential
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