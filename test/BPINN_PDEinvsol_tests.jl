using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
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

println("Example 3: 2D Periodic System with New parameter estimation")
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

discretization = BayesianPINN([chainl], GridTraining([0.02]), param_estim = true,
    dataset = [dataset, nothing])

# creating dictionary for masking equations
eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)],
    Dict_differentials = Dict_differentials, progress = true)

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])

@test u_predict≈u_real atol=1.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3

println("Example 3: Lotka Volterra with New parameter estimation")
@parameters t α β γ δ
@variables x(..) y(..)

Dt = Differential(t)
eqs = [Dt(x(t)) ~ α * x(t) - β * x(t) * y(t), Dt(y(t)) ~ -γ * y(t) + δ * x(t) * y(t)]
bcs = [x(0) ~ 1.0, y(0) ~ 1.0]
domains = [t ∈ Interval(0.0, 4.0)]

# Define the parameters' values
# params = [α => 1.0, β => 0.5, γ => 0.5, δ => 1.0]
# p = [1.5, 1.0, 3.0, 1.0]

chainl = [
    Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh),
        Lux.Dense(6, 1)),
    Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh),
        Lux.Dense(6, 1)),
]

initl, st = Lux.setup(Random.default_rng(), chainl[1])
initl1, st1 = Lux.setup(Random.default_rng(), chainl[2])

@named pde_system = PDESystem(eqs,
    bcs,
    domains,
    [t],
    [x(t), y(t)],
    [α, β, γ, δ],
    defaults = Dict([α => 1.0, β => 0.5, γ => 0.5, δ => 1.0]))

using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Distributions, Random

function lotka_volterra(u, p, t)
    α, β, γ, δ = p
    x, y = u
    dx = (α - β * y) * x
    dy = (δ * x - γ) * y
    return [dx, dy]
end

# initial-value problem.
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 4.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Solve using OrdinaryDiffEq.jl solver
dt = 0.05
solution = solve(prob, Tsit5(); saveat = dt)

# Extract solution
time = solution.t
u = hcat(solution.u...)
# plot(time, u[1, :])
# plot!(time, u[2, :])
# Construct dataset
dataset = [hcat(u[i, :], time) for i in 1:2]

discretization = BayesianPINN(chainl, GridTraining(0.01), param_estim = true,
    dataset = [dataset, nothing])

# creating dictionary for masking equations
eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

sol = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 500,
    bcstd = [0.05, 0.05],
    phystd = [0.005, 0.005], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(1.0, 2),
        Normal(1.0, 2),
        Normal(1.0, 2),
        Normal(1.0, 2),
    ], progress = true)

# plot!(sol.timepoints[1]', sol.ensemblesol[1])
# plot!(sol.timepoints[2]', sol.ensemblesol[2])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 500,
    bcstd = [0.05, 0.05],
    phystd = [0.005, 0.005], l2std = [0.1, 0.1],
    phystdnew = [0.1, 0.1],
    #  Kernel = AdvancedHMC.NUTS(0.8),
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 50.0],
    param = [
        Normal(1.0, 2),
        Normal(1.0, 2),
        Normal(1.0, 2),
        Normal(1.0, 2),
    ],
    Dict_differentials = Dict_differentials, progress = true)

# plot!(sol1.timepoints[1]', sol1.ensemblesol[1])
# plot!(sol1.timepoints[2]', sol1.ensemblesol[2])

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])

@test u_predict≈u_real atol=1.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3

# points1 = []
# for eq_arg in eq_args
#     a = []
#     # for each (depvar,[indvar1..]) if indvari==indvar (eq_arg)
#     for i in eachindex(symbols_input)
#         if symbols_input[i][2] == eq_arg
#             # include domain points of that depvar
#             # each loss equation take domain matrix [points..;points..]
#             push!(a, train_sets[i][:, 2:end]')
#         end
#     end
#     # vcat as new row for next equation
#     push!(points1, vcat(a...))
# end
# println(points1 == points)

using NeuralPDE, Flux, Lux, ModelingToolkit, LinearAlgebra, AdvancedHMC
import ModelingToolkit: Interval, infimum, supremum, Distributions
using Plots, MonteCarloMeasurements

@parameters x, t, α
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dx2 = Differential(x)^2
Dx3 = Differential(x)^3
Dx4 = Differential(x)^4

# α = 1
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
dx = 0.4;
dt = 0.2;

# Function to compute analytical solution at a specific point (x, t)
function u_analytic_point(x, t)
    z = -x / 2 + t
    return 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
end

# Function to generate the dataset matrix
function generate_dataset_matrix(domains, dx, dt)
    x_values = -10:dx:10
    t_values = 0.0:dt:1.0

    dataset = []

    for t in t_values
        for x in x_values
            u_value = u_analytic_point(x, t)
            push!(dataset, [u_value, x, t])
        end
    end

    return vcat([data' for data in dataset]...)
end

datasetpde = [generate_dataset_matrix(domains, dx, dt)]

# noise to dataset
noisydataset = deepcopy(datasetpde)
noisydataset[1][:, 1] = noisydataset[1][:, 1] .+
                        randn(size(noisydataset[1][:, 1])) .* 5 / 100 .*
                        noisydataset[1][:, 1]

# plot(datasetpde[1][:, 2], datasetpde[1][:, 1], title = "Dataset from Analytical Solution")
# plot!(noisydataset[1][:, 2], noisydataset[1][:, 1])

# Neural network
chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh),
    Lux.Dense(8, 8, Lux.tanh),
    Lux.Dense(8, 1))

discretization = NeuralPDE.BayesianPINN([chain],
    GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset, nothing])

@named pde_system = PDESystem(eq,
    bcs,
    domains,
    [x, t],
    [u(x, t)],
    [α],
    defaults = Dict([α => 0.5]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 100,
    bcstd = [0.2, 0.2, 0.2, 0.2, 0.2],
    phystd = [1.0], l2std = [0.05], param = [Distributions.LogNormal(0.5, 2)],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0], progress = true)

eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 100,
    bcstd = [0.2, 0.2, 0.2, 0.2, 0.2],
    phystd = [1.0], phystdnew = [0.05], l2std = [0.05],
    param = [Distributions.LogNormal(0.5, 2)],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0], Dict_differentials = Dict_differentials,
    progress = true)

phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, [dx / 10, dt])]
u_predict = [[first(pmean(phi([x, t], sol1.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

# p1 = plot(xs, u_predict, title = "predict")
# p2 = plot(xs, u_real, title = "analytic")
# p3 = plot(xs, diff_u, title = "error")
# plot(p1, p2, p3)

phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, [dx / 10, dt])]
u_predict = [[first(pmean(phi([x, t], sol2.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol2.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

# p1 = plot(xs, u_predict, title = "predict")
# p2 = plot(xs, u_real, title = "analytic")
# p3 = plot(xs, diff_u, title = "error")
# plot(p1, p2, p3)
