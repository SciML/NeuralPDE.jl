using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays, ModelingToolkit

Random.seed!(100)

@testset "Example 1: 1D Periodic System with parameter estimation" begin
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

    # TODO: correct BPINN implementations for Training Strategies.

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

    @test u_predict≈u_real atol=0.1
    @test mean(u_predict .- u_real) < 0.01
    @test sol1.estimated_de_params[1]≈param atol=0.1
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
            Lux.Dense(n, 1))
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
            Dict_differentials[eval(in_exp)] = Symbolics.variable("diff_$(length(Dict_differentials) + 1)")
            return
        else
            recur_expression(in_exp, Dict_differentials)
        end
    end
end

@testset "improvement in Solving Parametric Kuromo-Sivashinsky Equation" begin
    @parameters x, t, α
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    # α = 1 (KS equation to be parametric in a)
    β = 4
    γ = 1
    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) + α * Dx2(u(x, t)) + β * Dx3(u(x, t)) + γ * Dx4(u(x, t)) ~ 0

    u_analytic(x, t; z = -x / 2 + t) = 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
    du(x, t; z = -x / 2 + t) = 15 / 2 * (tanh(z) + 1) * (3 * tanh(z) - 1) * sech(z)^2

    bcs = [u(x, 0) ~ u_analytic(x, 0),
        u(-10, t) ~ u_analytic(-10, t),
        u(10, t) ~ u_analytic(10, t),
        Dx(u(-10, t)) ~ du(-10, t),
        Dx(u(10, t)) ~ du(10, t)]

    # Space and time domains
    domains = [x ∈ Interval(-10.0, 10.0),
        t ∈ Interval(0.0, 1.0)]

    # Discretization
    dx = 0.4
    dt = 0.2

    # Function to compute analytical solution at a specific point (x, t)
    function u_analytic_point(x, t)
        z = -x / 2 + t
        return 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
    end

    # Function to generate the dataset matrix
    function generate_dataset_matrix(domains, dx, dt, xlim, tlim)
        x_values = xlim[1]:dx:xlim[2]
        t_values = tlim[1]:dt:tlim[2]

        dataset = []

        for t in t_values
            for x in x_values
                u_value = u_analytic_point(x, t)
                push!(dataset, [u_value, x, t])
            end
        end

        return vcat([data' for data in dataset]...)
    end

    # considering sparse dataset from half of x's domain
    datasetpde_new = [generate_dataset_matrix(domains, dx, dt, [-10, 0], [0.0, 1.0])]

    # Adding Gaussian noise with a 0.8 std
    noisydataset_new = deepcopy(datasetpde_new)
    noisydataset_new[1][:, 1] = noisydataset_new[1][:, 1] .+
                                (randn(size(noisydataset_new[1][:, 1])) .* 0.8)

    # Neural network
    chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh),
        Lux.Dense(8, 8, Lux.tanh),
        Lux.Dense(8, 1))

    # Discretization for old and new models
    discretization = NeuralPDE.BayesianPINN([chain],
        GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset_new, nothing])

    # let α default to 2.0
    @named pde_system = PDESystem(eq,
        bcs,
        domains,
        [x, t],
        [u(x, t)],
        [α],
        defaults = Dict([α => 2.0]))

    # neccesarry for loss function construction (involves Operator masking) 
    eqs = pde_system.eqs
    Dict_differentials = Dict()
    exps = toexpr.(eqs)
    nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

    # Dict_differentials is now ;
    # Dict{Any, Any} with 5 entries:
    #   Differential(x)(Differential(x)(u(x, t)))            => diff_5
    #   Differential(x)(Differential(x)(Differential(x)(u(x… => diff_1
    #   Differential(x)(Differential(x)(Differential(x)(Dif… => diff_2
    #   Differential(x)(u(x, t))                             => diff_4
    #   Differential(t)(u(x, t))                             => diff_3

    # using HMC algorithm due to convergence, stability, time of training. (refer to mcmc chain plots)
    # choice of std for objectives is very important
    # pass in Dict_differentials, phystdnew arguments when using the new model

    sol_new = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 150,
        bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.2],
        phystd = [0.2], l2std = [0.5], param = [Distributions.Normal(2.0, 2)],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 100.0, 1 / 100.0],
        Dict_differentials = Dict_differentials)

    sol_old = ahmc_bayesian_pinn_pde(pde_system,
        discretization;
        draw_samples = 150,
        bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
        phystd = [0.2], l2std = [0.5], param = [Distributions.Normal(2.0, 2)],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 100.0, 1 / 100.0])

    phi = discretization.phi[1]
    xs, ts = [infimum(d.domain):dx:supremum(d.domain)
              for (d, dx) in zip(domains, [dx / 10, dt])]
    u_real = [[u_analytic(x, t) for x in xs] for t in ts]

    u_predict_new = [[first(pmean(phi([x, t], sol_new.estimated_nn_params[1]))) for x in xs]
                     for t in ts]

    diff_u_new = [[abs(u_analytic(x, t) -
                       first(pmean(phi([x, t], sol_new.estimated_nn_params[1]))))
                   for x in xs]
                  for t in ts]

    u_predict_old = [[first(pmean(phi([x, t], sol_old.estimated_nn_params[1]))) for x in xs]
                     for t in ts]
    diff_u_old = [[abs(u_analytic(x, t) -
                       first(pmean(phi([x, t], sol_old.estimated_nn_params[1]))))
                   for x in xs]
                  for t in ts]

    @test all(all, [((diff_u_new[i]) .^ 2 .< 0.5) for i in 1:6]) == true
    @test all(all, [((diff_u_old[i]) .^ 2 .< 0.5) for i in 1:6]) == false

    MSE_new = [sum(abs2, diff_u_new[i]) for i in 1:6]
    MSE_old = [sum(abs2, diff_u_old[i]) for i in 1:6]
    @test (MSE_new .< MSE_old) == [1, 1, 1, 1, 1, 1]

    param_new = sol_new.estimated_de_params[1]
    param_old = sol_old.estimated_de_params[1]
    α = 1
    @test abs(param_new - α) < 0.2 * α
    @test abs(param_new - α) < abs(param_old - α)
end