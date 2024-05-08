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

    # TODO: correct implementations
    # # checking all training strategies
    # discretization = BayesianPINN([chainl], StochasticTraining(200), param_estim = true,
    #     dataset = [dataset, nothing])

    # ahmc_bayesian_pinn_pde(pde_system,
    #     discretization;
    #     draw_samples = 1500,
    #     bcstd = [0.05],
    #     phystd = [0.01], l2std = [0.01],
    #     priorsNNw = (0.0, 1.0),
    #     saveats = [1 / 50.0],
    #     param = [LogNormal(6.0, 0.5)])

    # discretization = BayesianPINN([chainl], QuasiRandomTraining(200), param_estim = true,
    #     dataset = [dataset, nothing])

    # ahmc_bayesian_pinn_pde(pde_system,
    #     discretization;
    #     draw_samples = 1500,
    #     bcstd = [0.05],
    #     phystd = [0.01], l2std = [0.01],
    #     priorsNNw = (0.0, 1.0),
    #     saveats = [1 / 50.0],
    #     param = [LogNormal(6.0, 0.5)])

    # discretization = BayesianPINN([chainl], QuadratureTraining(), param_estim = true,
    #     dataset = [dataset, nothing])

    # ahmc_bayesian_pinn_pde(pde_system,
    #     discretization;
    #     draw_samples = 1500,
    #     bcstd = [0.05],
    #     phystd = [0.01], l2std = [0.01],
    #     priorsNNw = (0.0, 1.0),
    #     saveats = [1 / 50.0],
    #     param = [LogNormal(6.0, 0.5)])

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

println("Example 3: 2D Periodic System with New parameter estimation")
@parameters t, p
@variables u(..)

Dt = Differential(t)
eqs = Dt(u(t)) - cos(p * t) * u(t) ~ 0
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
    phystd = [0.01], l2std = [0.01], phystdnew = [0.05],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(6.0, 0.5)],
    Dict_differentials = Dict_differentials)

sol2 = ahmc_bayesian_pinn_pde(pde_system,
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

ts = vec(sol2.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol2.ensemblesol[1])

@test u_predict≈u_real atol=1.5
@test mean(u_predict .- u_real) < 0.1
@test sol1.estimated_de_params[1]≈param atol=param * 0.3


println("Example 3: Lotka Volterra with New parameter estimation")
@parameters t α β γ δ
@variables x(..) y(..)

Dt = Differential(t)
eqs = [Dt(x(t)) * α  ~ x(t) - β * x(t) * y(t), Dt(y(t)) * δ ~ x(t) * y(t) - y(t)*γ ]
bcs = [x(0) ~ 1.0, y(0) ~ 1.0]
domains = [t ∈ Interval(0.0, 7.0)]

# Define the parameters' values
# α, β, γ, δ = p

# regular equations
# dx = (1.5 - y) * x # prey
# dy = (x - 3.0) * y # predator
# p = [1.5, 1.0, 3.0, 1.0] non transformed values

# transformed equations
# dx*0.666 = (1 - 0.666 * y) * x # prey
# dy*1.0 = (x - 3.0) * y # predator
# p = [0.666, 0.666, 3.0, 1.0] transformed values (change is scale also ensured!)

chainl = [
    Lux.Chain(Lux.Dense(1, 5, sin), Lux.Dense(5, 5, sin),Lux.Dense(5, 1)),
    Lux.Chain(Lux.Dense(1, 5, sin), Lux.Dense(5, 5, sin),Lux.Dense(5, 1))
]

initl, st = Lux.setup(Random.default_rng(), chainl[1])
initl1, st1 = Lux.setup(Random.default_rng(), chainl[2])

using NeuralPDE, Lux, OrdinaryDiffEq, Distributions, Random

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
# initial-value problem.
u0 = [1.0, 1.0]
# p = [2/3, 2/3, 1/3.0, 1/3.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 7.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)
dt = 0.01
solution = solve(prob, Tsit5(); saveat = dt)
solution1 = solve(prob, Tsit5(); saveat = 0.02)

function calculate_errors(approx_sol, solution_points)
  # Check vector lengths match
  if length(approx_sol) != length(solution_points)
    error("Vectors must have the same length")
  end

  # Calculate errors
  n = length(approx_sol)
  errors = randn(n)
  for i in 1:n
    errors[i] = solution_points[i] - approx_sol[i]
  end

  # Calculate RMSE
  rmse = sqrt(mean(errors.^2))

  # Calculate MAE
  mae = mean(abs.(errors))

  # Calculate maximum absolute error
  max_error = maximum(abs.(errors))

  # Return dictionary with errors
  return Dict(
      "RMSE" => rmse,
      "MAE" => mae,
      "Max Abs Error" => max_error,
  )
end
u = hcat(solution1.u...)
u[1,:]
sol6_2.ensemblesol[1]

a1=calculate_errors(pmean(sol6_1.ensemblesol[1]), u1[1,:])
b1=calculate_errors(pmean(sol6_1.ensemblesol[2]), u1[2,:])

a=calculate_errors(pmean(sol6_2.ensemblesol[1]), u[1,:])
b=calculate_errors(pmean(sol6_2.ensemblesol[2]), u[2,:])

c=calculate_errors(pmean(sol6_L2_2.ensemblesol[1]), u[1,:])
d=calculate_errors(pmean(sol6_L2_2.ensemblesol[2]), u[2,:])

e=calculate_errors(pmean(sol6_L2_1.ensemblesol[1]), u[1,:])
f=calculate_errors(pmean(sol6_L2_1.ensemblesol[2]), u[2,:])

g=calculate_errors(pmean(sol6_L2.ensemblesol[1]), u[1,:])
h=calculate_errors(pmean(sol6_L2.ensemblesol[2]), u[2,:])
sol6_2.ensemblesol[1]
sol6_2.ensemblesol[2]

sol6_L2.ensemblesol[1]
sol6_L2.ensemblesol[2]

# function moving_average_smoothing(data::Vector{T}, window_size::Int) where {T}
#     smoothed_data = similar(data, T, length(data))

#     for i in 1:length(data)
#         start_idx = max(1, i - window_size)
#         end_idx = min(length(data), i + window_size)
#         smoothed_data[i] = mean(data[start_idx:end_idx])
#     end

#     return smoothed_data'
# end

# Extract solution
time = solution.t
u = hcat(solution.u...)
time1=solution.t
u_noisy = u .+ u .* (0.2 .* randn(size(u)))
u_noisy0 = u .+ (3.0 .* rand(size(u)[1],size(u)[2]) .- 1.5)
u_noisy1 = u .+ (0.8.* randn(size(Array(solution))))
u_noisy2 = u .+ (0.5.* randn(size(Array(solution))))

plot(time,u[1,:])
plot!(time,u[2,:])
scatter!(time1,u_noisy0[1,:])
scatter!(time1,u_noisy0[2,:])
scatter!(discretization_08_gaussian.dataset[1][1][:,2], discretization_08_gaussian.dataset[1][1][:,1])
scatter!(discretization_08_gaussian.dataset[1][2][:,2], discretization_08_gaussian.dataset[1][2][:,1])

scatter!(discretization_05_gaussian.dataset[1][1][:,2], discretization_05_gaussian.dataset[1][1][:,1])
scatter!(discretization_05_gaussian.dataset[1][2][:,2], discretization_05_gaussian.dataset[1][2][:,1])
# discretization_05_gaussian.dataset[1][1][:,2]
# window_size = 5
# smoothed_datasets = [moving_average_smoothing(u1[i, :], window_size)
#                      for i in 1:length(solution.u[1])]
# u2 = vcat(smoothed_datasets[1], smoothed_datasets[2])
# Randomly select some points from the solution
num_points = 100  # Number of points to select
selected_indices = rand(1:size(u_noisy1, 2), num_points)
upoints = [u_noisy1[:, i] for i in selected_indices]
timepoints = [time[i] for i in selected_indices]
temp=hcat(upoints...)
dataset = [hcat(temp[i, :], timepoints) for i in 1:2]

discretization_uniform = BayesianPINN(chainl, GridTraining([0.01]), param_estim = true,
    dataset = [dataset, nothing])
discretization_08_gaussian = BayesianPINN(chainl, GridTraining([0.01]), param_estim = true,
    dataset = [dataset, nothing])
discretization_05_gaussian = BayesianPINN(chainl, GridTraining([0.01]), param_estim = true,
    dataset = [dataset, nothing])

discretization1 = BayesianPINN(chainl, GridTraining([0.01]), param_estim = true,
    dataset = [dataset, nothing])

scatter!(discretization.dataset[1][1][:,2], discretization.dataset[1][1][:,1])
scatter!(discretization.dataset[1][2][:,2], discretization.dataset[1][2][:,1])

sol = solve(prob, Tsit5(); saveat=0.1)
odedata = Array(sol) + 0.8 * randn(size(Array(sol)))

@named pde_system = PDESystem(eqs,
    bcs,
    domains,
    [t],
    [x(t), y(t)],
    [α, β, γ, δ],
    defaults = Dict([α =>2, β => 2, γ =>2, δ =>2]))

# creating dictionary for masking equations
eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

sol3 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_uniform = ahmc_bayesian_pinn_pde(pde_system,
    discretization_uniform;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_08_gaussian = ahmc_bayesian_pinn_pde(pde_system,
    discretization_08_gaussian;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_05_gaussian = ahmc_bayesian_pinn_pde(pde_system,
    discretization_05_gaussian;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)


# more iterations for above
sol3_100_uniform_1000 = ahmc_bayesian_pinn_pde(pde_system,
    discretization_uniform;
    draw_samples = 1000,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_08_gaussian_1000 = ahmc_bayesian_pinn_pde(pde_system,
    discretization_08_gaussian;
    draw_samples = 1000,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_05_gaussian_1000 = ahmc_bayesian_pinn_pde(pde_system,
    discretization_05_gaussian;
    draw_samples = 1000,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

    
# more iterations for above + strict BC
sol3_100_uniform_1000_bc = ahmc_bayesian_pinn_pde(pde_system,
    discretization_uniform;
    draw_samples = 1000,
    bcstd = [0.1, 0.1],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_08_gaussian_1000_bc = ahmc_bayesian_pinn_pde(pde_system,
    discretization_08_gaussian;
    draw_samples = 1000,
    bcstd = [0.1, 0.1],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_08_gaussian_1000_bc_hard = ahmc_bayesian_pinn_pde(pde_system,
    discretization_08_gaussian;
    draw_samples = 1000,
    bcstd = [0.05, 0.05],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_05_gaussian_1000_bc = ahmc_bayesian_pinn_pde(pde_system,
    discretization_05_gaussian;
    draw_samples = 1000,
    bcstd = [0.1, 0.1],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol3_100_08_gaussian_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_08_gaussian;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol3_100_05_gaussian_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_05_gaussian;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

# time
# dataset
# chainl[1](time', sol3.estimated_nn_params[1], st)[1][1,:]
# plot!(time1, chainl[1](time1', sol3.estimated_nn_params[1], st)[1][1,:])
# plot!(time1, chainl[2](time1', sol3.estimated_nn_params[2], st)[1][1,:])
# plot!(time1, chainl[1](time1', sol5.estimated_nn_params[1], st)[1][1,:])
# plot!(time1, chainl[2](time1', sol5.estimated_nn_params[2], st)[1][1,:])
# time1 = collect(0.0:(1 / 100.0):8.0)

sol4 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true
)

sol4_0 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true
)


sol5_00 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol5_0 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.2, 0.2],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol5 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)


# 70 points in dataset
sol6 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.2, 0.2],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

# SOL6_1 VS SOL6_L2
sol6_1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol6_2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol6_2_L2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol6_3 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol6_4 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.2, 0.2],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol6_L2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)
    
sol6_L2_1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

sol6_L2_2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.05, 0.05],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true)

# 50 datapoint 0-5 sol5 vs sol4
# julia> sol4.estimated_de_params
# 4-element Vector{Particles{Float64, 234}}:
#  0.549 ± 0.0058
#  0.71 ± 0.0042
#  0.408 ± 0.0063
#  0.355 ± 0.0015

# julia> sol5.estimated_de_params
# 4-element Vector{Particles{Float64, 234}}:
#  0.604 ± 0.0052
#  0.702 ± 0.0034
#  0.346 ± 0.0037
#  0.335 ± 0.0013

# 100 datapoint 0-5 sol5_2 vs sol3
# julia> sol3.estimated_de_params
# 4-element Vector{Particles{Float64, 234}}:
#  0.598 ± 0.0037
#  0.711 ± 0.0027
#  0.399 ± 0.0032
#  0.333 ± 0.0011

# julia> sol5_2.estimated_de_params
# 4-element Vector{Particles{Float64, 234}}:
#  0.604 ± 0.0035
#  0.686 ± 0.0026
#  0.395 ± 0.0029
#  0.328 ± 0.00095

# timespan for full dataset (0-8)
sol6 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(1, 2),
        Normal(1, 1),
        Normal(1, 2),
        Normal(1, 1)
    ], progress = true)

sol5_3 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(1, 2),
        Normal(1, 1),
        Normal(1, 2),
        Normal(1, 1)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol5_4 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.2, 0.2],
    saveats = [1 / 50.0],
    param = [
        Normal(1, 2),
        Normal(1, 1),
        Normal(1, 2),
        Normal(1, 1)
    ], Dict_differentials = Dict_differentials, progress = true
)

sol5_5 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(1, 2),
        Normal(1, 1),
        Normal(1, 2),
        Normal(1, 1)
    ], progress = true
)

sol7 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(1, 2),
        Normal(1, 1),
        Normal(1, 2),
        Normal(1, 1)
    ], Dict_differentials = Dict_differentials, progress = true)

sol5_5_1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 1),
        Normal(2, 2),
        Normal(2, 1)
    ], progress = true
)

sol7_1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 1),
        Normal(2, 2),
        Normal(2, 1)
    ], Dict_differentials = Dict_differentials, progress = true)
    
sol7_2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.1, 0.1],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 1),
        Normal(2, 2),
        Normal(2, 1)
    ], Dict_differentials = Dict_differentials, progress = true)

sol7_3 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.2, 0.2],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 1),
        Normal(2, 2),
        Normal(2, 1)
    ], Dict_differentials = Dict_differentials, progress = true)
 
sol7_4 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 1),
        Normal(2, 2),
        Normal(2, 1)
    ], Dict_differentials = Dict_differentials, progress = true)

lpfun = function f(chain::Chains) # function to compute the logpdf values
    niter, nparams, nchains = size(chain)
    lp = zeros(niter + nchains) # resulting logpdf values
    for i = 1:nparams
        lp += logpdf(MvNormal(Array(chain[:,i,:])) , dataset[1][:,1]')
        lp += logpdf(MvNormal(Array(chain[:,i,:])) , dataset[1][:,2]')
    end
    return lp
end

DIC, pD = dic(sol3.original.mcmc_chain, lpfun)
DIC1, pD1 = dic(sol4.original.mcmc_chain, lpfun)

size(sol3.original.mcmc_chain)
Array(sol3.original.mcmc_chain[1,:,:])
length(sol3.estimated_nn_params[1])
chainl[1](time', sol3.estimated_nn_params[1], st)[1]

data = [hcat(calculate_derivatives2(dataset[i][:, 2], dataset[1][:, 1]),dataset[i][:, 2]) for i in eachindex(dataset)]
dataset[1][:,1]
dataset[2]
plot!(dataset[1][:,2],dataset[1][:,1])
eqs
sol5 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 200,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.02, 0.02],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(3, 2),
        Normal(3, 2)
        # LogNormal(1, 2),
        # LogNormal(1, 2),
        # LogNormal(1, 2),
        # LogNormal(1, 2)
    ], progress = true)

# plot(time, chainl[1](time', sol2.estimated_nn_params[1], st)[1])
# plot!(time, chainl[2](time', sol2.estimated_nn_params[2], st)[1])

sol6 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 200,
    bcstd = [0.5, 0.5],
    phystd = [0.5, 0.5], l2std = [0.02, 0.02],
    priorsNNw = (0.0, 5.0), phystdnew = [0.5, 0.5],
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),aa
    param = [
        # LogNormal(2, 3),LogNormal(2, 3),LogNormal(2, 3),LogNormal(2, 3)
        # Normal(3, 2),
        # Normal(4, 2),
        Normal(3, 2),
        Normal(3, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

function calculate_derivatives2(indvar,depvar)
    x̂, time = indvar,depvar
    num_points = length(x̂)
    # Initialize an array to store the derivative values.
    derivatives = similar(x̂)

    for i in 2:(num_points - 1)
        # Calculate the first-order derivative using central differences.
        Δt_forward = time[i + 1] - time[i]
        Δt_backward = time[i] - time[i - 1]

        derivative = (x̂[i + 1] - x̂[i - 1]) / (Δt_forward + Δt_backward)

        derivatives[i] = derivative
    end

    # Derivatives at the endpoints can be calculated using forward or backward differences.
    derivatives[1] = (x̂[2] - x̂[1]) / (time[2] - time[1])
    derivatives[end] = (x̂[end] - x̂[end - 1]) / (time[end] - time[end - 1])
    return derivatives
end
dataset[1]
dataset[2]
dataset[1][:,1]=calculate_derivatives2(dataset[1][:,2], dataset[1][:,1])
dataset[2][:,1]=calculate_derivatives2(dataset[2][:,2], dataset[2][:,1])
dataset[1]
dataset[2]
sol7 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 200,
    bcstd = [0.5, 0.5],
    phystd = [0.5, 0.5], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),
    param = [
        Normal(0, 2),
        Normal(0, 2)
        # LogNormal(1, 2),
        # LogNormal(1, 2),
        # LogNormal(1, 2),
        # LogNormal(1, 2)
    ], progress = true)

# plot(time, chainl[1](time', sol2.estimated_nn_params[1], st)[1])
# plot!(time, chainl[2](time', sol2.estimated_nn_params[2], st)[1])

sol8 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.1, 0.1],
    phystd = [0.1, 0.1], l2std = [0.1, 0.1],
    priorsNNw = (0.0, 5.0), phystdnew = [0.1, 0.1],
    saveats = [1 / 50.0],
    # Kernel = AdvancedHMC.NUTS(0.8),aa
    param = [
        # LogNormal(2, 3),LogNormal(2, 3),LogNormal(2, 3),LogNormal(2, 3)
        # Normal(3, 2),
        # Normal(4, 2),
        Normal(0, 2),
        Normal(0, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

timepoints = collect(0.0:(1 / 100.0):9.0)
plot!(timepoints', chainl[1](timepoints', sol5_4.estimated_nn_params[1], st)[1])
plot!(timepoints, chainl[2](timepoints', sol5_4.estimated_nn_params[2], st)[1])

sol_L2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true
)

sol_NEW = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)
sol_L2_70 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0),
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], progress = true
)

sol_NEW_70 = ahmc_bayesian_pinn_pde(pde_system,
    discretization1;
    draw_samples = 700,
    bcstd = [0.2, 0.2],
    phystd = [0.2, 0.2], l2std = [0.05, 0.05],
    priorsNNw = (0.0, 5.0), 
    phystdnew = [0.3, 0.3],
    saveats = [1 / 50.0],
    param = [
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(2, 2)
    ], Dict_differentials = Dict_differentials, progress = true
)

using Plots, StatsPlots
plotly()

plot(time, u[1, :])
plot!(time, u[2, :])
scatter!(time, u_noisy[1, :])
scatter!(time, u_noisy[2, :])
scatter!(discretization.dataset[1][1][:,2], discretization.dataset[1][1][:,1])
scatter!(discretization.dataset[1][2][:,2], discretization.dataset[1][2][:,1])

scatter!(discretization1.dataset[1][1][:,2], discretization1.dataset[1][1][:,1],legend=nothing)
scatter!(discretization1.dataset[1][2][:,2], discretization1.dataset[1][2][:,1])

# plot28(sol4 seems better vs sol3 plots, params seems similar)
plot!(sol3.timepoints[1]', sol3.ensemblesol[1])
plot!(sol3.timepoints[2]', sol3.ensemblesol[2])
plot!(sol3_0.timepoints[1]', sol3_0.ensemblesol[1])
plot!(sol3_0.timepoints[2]', sol3_0.ensemblesol[2])

plot!(sol4.timepoints[1]', sol4.ensemblesol[1])
plot!(sol4.timepoints[2]', sol4.ensemblesol[2])
plot!(sol4_0.timepoints[1]', sol4_0.ensemblesol[1])
plot!(sol4_0.timepoints[2]', sol4_0.ensemblesol[2])

plot!(sol4_2.timepoints[1]', sol4_2.ensemblesol[1],legend=nothing)
plot!(sol4_2.timepoints[2]', sol4_2.ensemblesol[2])
plot!(sol5_2.timepoints[1]', sol5_2.ensemblesol[1],legend=nothing)
plot!(sol5_2.timepoints[2]', sol5_2.ensemblesol[2])

plot!(sol4_3.timepoints[1]', sol4_3.ensemblesol[1],legend=nothing)
plot!(sol4_3.timepoints[2]', sol4_3.ensemblesol[2])
plot!(sol5_3.timepoints[1]', sol5_3.ensemblesol[1])
plot!(sol5_3.timepoints[2]', sol5_3.ensemblesol[2])
plot!(sol5_4.timepoints[1]', sol5_4.ensemblesol[1],legend=nothing)
plot!(sol5_4.timepoints[2]', sol5_4.ensemblesol[2])


# plot 36 sol4 vs sol5(params sol4 better, but plots sol5 "looks" better),plot 44(sol5 better than sol6 overall)
plot!(sol5.timepoints[1]', sol5.ensemblesol[1],legend=nothing)
plot!(sol5.timepoints[2]', sol5.ensemblesol[2])

plot!(sol5_0.timepoints[1]', sol5_0.ensemblesol[1],legend=nothing)
plot!(sol5_0.timepoints[2]', sol5_0.ensemblesol[2])

plot!(sol5_00.timepoints[1]', sol5_00.ensemblesol[1],legend=nothing)
plot!(sol5_00.timepoints[2]', sol5_00.ensemblesol[2])

plot!(sol6.timepoints[1]', sol6.ensemblesol[1])
plot!(sol6.timepoints[2]', sol6.ensemblesol[2])
plot!(sol6_L2.timepoints[1]', sol6_L2.ensemblesol[1])
plot!(sol6_L2.timepoints[2]', sol6_L2.ensemblesol[2])

plot!(sol6_L2_1.timepoints[1]', sol6_L2_1.ensemblesol[1])
plot!(sol6_L2_1.timepoints[2]', sol6_L2_1.ensemblesol[2])

plot!(sol6_L2_2.timepoints[1]', sol6_L2_2.ensemblesol[1])
plot!(sol6_L2_2.timepoints[2]', sol6_L2_2.ensemblesol[2])

plot!(sol6_1.timepoints[1]', sol6_1.ensemblesol[1])
plot!(sol6_1.timepoints[2]', sol6_1.ensemblesol[2])
plot!(sol6_2.timepoints[1]', sol6_2.ensemblesol[1])
plot!(sol6_2.timepoints[2]', sol6_2.ensemblesol[2],legend=nothing)
plot!(sol6_2_L2.timepoints[1]', sol6_2_L2.ensemblesol[1])
plot!(sol6_2_L2.timepoints[2]', sol6_2_L2.ensemblesol[2],legend=nothing)

# plot52 sol7 vs sol5(sol5 overall better plots, params?)
plot!(sol7.timepoints[1]', sol7.ensemblesol[1])
plot!(sol7.timepoints[2]', sol7.ensemblesol[2])

# sol8,sol8_2,sol9,sol9_2 bad
plot!(sol8.timepoints[1]', sol8.ensemblesol[1])
plot!(sol8.timepoints[2]', sol8.ensemblesol[2])
plot!(sol8_2.timepoints[1]', sol8_2.ensemblesol[1])
plot!(sol8_2.timepoints[2]', sol8_2.ensemblesol[2])

plot!(sol9.timepoints[1]', sol9.ensemblesol[1])
plot!(sol9.timepoints[2]', sol9.ensemblesol[2])
plot!(sol9_2.timepoints[1]', sol9_2.ensemblesol[1])
plot!(sol9_2.timepoints[2]', sol9_2.ensemblesol[2])


plot!(sol5_5.timepoints[1]', sol5_5.ensemblesol[1])
plot!(sol5_5.timepoints[2]', sol5_5.ensemblesol[2],legend=nothing)

plot!(sol5_5_1.timepoints[1]', sol5_5_1.ensemblesol[1])
plot!(sol5_5_1.timepoints[2]', sol5_5_1.ensemblesol[2],legend=nothing)
plot!(sol7_1.timepoints[1]', sol7_1.ensemblesol[1])
plot!(sol7_1.timepoints[2]', sol7_1.ensemblesol[2])

plot!(sol7_4.timepoints[1]', sol7_4.ensemblesol[1])
plot!(sol7_4.timepoints[2]', sol7_4.ensemblesol[2])

plot!(sol5_2_1.timepoints[1]', sol5_2_1.ensemblesol[1],legend=nothing)
plot!(sol5_2_1.timepoints[2]', sol5_2_1.ensemblesol[2])
plot!(sol5_2_2.timepoints[1]', sol5_2_2.ensemblesol[1],legend=nothing)
plot!(sol5_2_2.timepoints[2]', sol5_2_2.ensemblesol[2])

plot!(sol5_0.timepoints[1]', sol5_0.ensemblesol[1])
plot!(sol5_0.timepoints[2]', sol5_0.ensemblesol[2],legend=nothing)

plot!(sol5_00.timepoints[1]', sol5_00.ensemblesol[1],legend=nothing)
plot!(sol5_00.timepoints[2]', sol5_00.ensemblesol[2])

plot!(sol3_0.timepoints[1]', sol3_0.ensemblesol[1])
plot!(sol3_0.timepoints[2]', sol3_0.ensemblesol[2],legend=nothing)
plot!(sol4_0.timepoints[1]', sol4_0.ensemblesol[1])
plot!(sol4_0.timepoints[2]', sol4_0.ensemblesol[2],legend=nothing)

plot!(sol3_100_05_gaussian.timepoints[1]', sol3_100_05_gaussian.ensemblesol[1])
plot!(sol3_100_05_gaussian.timepoints[2]', sol3_100_05_gaussian.ensemblesol[2],legend=nothing)

plot!(sol3_100_05_gaussian_new.timepoints[1]', sol3_100_05_gaussian_new.ensemblesol[1])
plot!(sol3_100_05_gaussian_new.timepoints[2]', sol3_100_05_gaussian_new.ensemblesol[2])

plot!(sol3_100_08_gaussian.timepoints[1]', sol3_100_08_gaussian.ensemblesol[1])
plot!(sol3_100_08_gaussian.timepoints[2]', sol3_100_08_gaussian.ensemblesol[2])

plot!(sol3_100_08_gaussian_new.timepoints[1]', sol3_100_08_gaussian_new.ensemblesol[1])
plot!(sol3_100_08_gaussian_new.timepoints[2]', sol3_100_08_gaussian_new.ensemblesol[2],legend=nothing)

plot!(sol3_100_uniform.timepoints[1]', sol3_100_uniform.ensemblesol[1])
plot!(sol3_100_uniform.timepoints[2]', sol3_100_uniform.ensemblesol[2])

plot!(sol3_100_08_gaussian_1000.timepoints[1]', sol3_100_08_gaussian_1000.ensemblesol[1])
plot!(sol3_100_08_gaussian_1000.timepoints[2]', sol3_100_08_gaussian_1000.ensemblesol[2])

plot!(sol3_100_05_gaussian_1000.timepoints[1]', sol3_100_05_gaussian_1000.ensemblesol[1])
plot!(sol3_100_05_gaussian_1000.timepoints[2]', sol3_100_05_gaussian_1000.ensemblesol[2])

plot!(sol3_100_uniform_1000.timepoints[1]', sol3_100_uniform_1000.ensemblesol[1])
plot!(sol3_100_uniform_1000.timepoints[2]', sol3_100_uniform_1000.ensemblesol[2])

plot!(sol3_100_08_gaussian_1000_bc.timepoints[1]', sol3_100_08_gaussian_1000_bc.ensemblesol[1])
plot!(sol3_100_08_gaussian_1000_bc.timepoints[2]', sol3_100_08_gaussian_1000_bc.ensemblesol[2])

# test with lower number of points
# test same calls 2 times or more
# consider full range dataset case
# combination of all above

# run 1 100 iters
sol5.estimated_de_params
sol6.estimated_de_params

# run 2 200 iters
sol5.estimated_de_params
sol6.estimated_de_params

# run 2 200 iters
sol3.estimated_de_params
sol4.estimated_de_params

# p = [2/3, 2/3, 1/3, 1/3]
sol3.estimated_de_params
sol4.estimated_de_params
dataset[1]
eqs
α, β, γ, δ = p
p
#  1.0
#  0.6666666666666666
#  1.0
#  0.33333333333333333

1/a
1/c
eqs
using StatsPlots
plotly()
plot(sol3.original.mcmc_chain)
plot(sol5_00.original.mcmc_chain)

# 4-element Vector{Particles{Float64, 34}}:
#  1.23 ± 0.022
#  0.858 ± 0.011
#  3.04 ± 0.079
#  1.03 ± 0.024
# 4-element Vector{Particles{Float64, 34}}:
#  1.2 ± 0.0069
#  0.835 ± 0.006
#  3.22 ± 0.01
#  1.08 ± 0.0053
# # plot(time', chainl[1](time', sol1.estimated_nn_params[1], st)[1])
# # plot!(time, chainl[2](time', sol1.estimated_nn_params[2], st)[1])

# sol3 = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 500,
#     bcstd = [0.05, 0.05],
#     phystd = [0.005, 0.005], l2std = [0.1, 0.1],
#     phystdnew = [0.5, 0.5],
#     #  Kernel = AdvancedHMC.NUTS(0.8),
#     priorsNNw = (0.0, 10.0),
#     saveats = [1 / 50.0],
#     param = [
#         Normal(0.0, 2),
#         Normal(0.0, 2),
#         Normal(0.0, 2),
#         Normal(0.0, 2)
#     ],
#     Dict_differentials = Dict_differentials, progress = true)

# sol = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 500,
#     bcstd = [0.05, 0.05],
#     phystd = [0.005, 0.005], l2std = [0.1, 0.1],
#     priorsNNw = (0.0, 10.0),
#     saveats = [1 / 50.0],
#     # Kernel = AdvancedHMC.NUTS(0.8),
#     param = [
#         Normal(1.0, 2),
#         Normal(1.0, 2),
#         Normal(1.0, 2),
#         Normal(1.0, 2)
#     ], progress = true)

# plot!(sol.timepoints[1]', sol.ensemblesol[1])
# plot!(sol.timepoints[2]', sol.ensemblesol[2])

# sol1 = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 500,
#     bcstd = [0.05, 0.05],
#     phystd = [0.005, 0.005], l2std = [0.1, 0.1],
#     phystdnew = [0.5, 0.5],
#     #  Kernel = AdvancedHMC.NUTS(0.8),
#     priorsNNw = (0.0, 10.0),
#     saveats = [1 / 50.0],
#     param = [
#         Normal(1.0, 2),
#         Normal(1.0, 2),
#         Normal(1.0, 2),
#         Normal(1.0, 2)
#     ],
#     Dict_differentials = Dict_differentials, progress = true)

# plot!(sol1.timepoints[1]', sol1.ensemblesol[1])
# plot!(sol1.timepoints[2]', sol1.ensemblesol[2])

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
        Normal(1.0, 2)
    ])

# plot!(sol.timepoints[1]', sol.ensemblesol[1])
# plot!(sol.timepoints[2]', sol.ensemblesol[2])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 500,
    bcstd = [0.05, 0.05],
    phystd = [0.005, 0.005], l2std = [0.1, 0.1],
    phystdnew = [0.5, 0.5],
    #  Kernel = AdvancedHMC.NUTS(0.8),
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 50.0],
    param = [
        Normal(1.0, 2),
        Normal(1.0, 2),
        Normal(1.0, 2),
        Normal(1.0, 2)
    ],
    Dict_differentials = Dict_differentials)

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

# using NeuralPDE, Flux, Lux, ModelingToolkit, LinearAlgebra, AdvancedHMC
# import ModelingToolkit: Interval, infimum, supremum, Distributions
# using Plots, MonteCarloMeasurements

# @parameters x, t, α
# @variables u(..)
# Dt = Differential(t)
# Dx = Differential(x)
# Dx2 = Differential(x)^2
# Dx3 = Differential(x)^3
# Dx4 = Differential(x)^4

# # α = 1
# β = 4
# γ = 1
# eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) + α * Dx2(u(x, t)) + β * Dx3(u(x, t)) + γ * Dx4(u(x, t)) ~ 0

# u_analytic(x, t; z = -x / 2 + t) = 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
# du(x, t; z = -x / 2 + t) = 15 / 2 * (tanh(z) + 1) * (3 * tanh(z) - 1) * sech(z)^2

# bcs = [u(x, 0) ~ u_analytic(x, 0),
#     u(-10, t) ~ u_analytic(-10, t),
#     u(10, t) ~ u_analytic(10, t),
#     Dx(u(-10, t)) ~ du(-10, t),
#     Dx(u(10, t)) ~ du(10, t)]

# # Space and time domains
# domains = [x ∈ Interval(-10.0, 10.0),
#     t ∈ Interval(0.0, 1.0)]

# # Discretization
# dx = 0.4;
# dt = 0.2;

# # Function to compute analytical solution at a specific point (x, t)
# function u_analytic_point(x, t)
#     z = -x / 2 + t
#     return 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
# end

# # Function to generate the dataset matrix
# function generate_dataset_matrix(domains, dx, dt)
#     x_values = -10:dx:10
#     t_values = 0.0:dt:1.0

#     dataset = []

#     for t in t_values
#         for x in x_values
#             u_value = u_analytic_point(x, t)
#             push!(dataset, [u_value, x, t])
#         end
#     end

#     return vcat([data' for data in dataset]...)
# end

# datasetpde = [generate_dataset_matrix(domains, dx, dt)]

# # noise to dataset
# noisydataset = deepcopy(datasetpde)
# noisydataset[1][:, 1] = noisydataset[1][:, 1] .+
#                         randn(size(noisydataset[1][:, 1])) .* 5 / 100 .*
#                         noisydataset[1][:, 1]

# # plot(datasetpde[1][:, 2], datasetpde[1][:, 1], title = "Dataset from Analytical Solution")
# # plot!(noisydataset[1][:, 2], noisydataset[1][:, 1])

# # Neural network
# chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh),
#     Lux.Dense(8, 8, Lux.tanh),
#     Lux.Dense(8, 1))

# discretization = NeuralPDE.BayesianPINN([chain],
#     GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset, nothing])

# @named pde_system = PDESystem(eq,
#     bcs,
#     domains,
#     [x, t],
#     [u(x, t)],
#     [α],
#     defaults = Dict([α => 0.5]))

# sol1 = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 100,
#     bcstd = [0.2, 0.2, 0.2, 0.2, 0.2],
#     phystd = [1.0], l2std = [0.05], param = [Distributions.LogNormal(0.5, 2)],
#     priorsNNw = (0.0, 10.0),
#     saveats = [1 / 100.0, 1 / 100.0], progress = true)

# eqs = pde_system.eqs
# Dict_differentials = Dict()
# exps = toexpr.(eqs)
# nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

# sol2 = ahmc_bayesian_pinn_pde(pde_system,
#     discretization;
#     draw_samples = 100,
#     bcstd = [0.2, 0.2, 0.2, 0.2, 0.2],
#     phystd = [1.0], phystdnew = [0.05], l2std = [0.05],
#     param = [Distributions.LogNormal(0.5, 2)],
#     priorsNNw = (0.0, 10.0),
#     saveats = [1 / 100.0, 1 / 100.0], Dict_differentials = Dict_differentials,
#     progress = true)

# phi = discretization.phi[1]
# xs, ts = [infimum(d.domain):dx:supremum(d.domain)
#           for (d, dx) in zip(domains, [dx / 10, dt])]
# u_predict = [[first(pmean(phi([x, t], sol1.estimated_nn_params[1]))) for x in xs]
#              for t in ts]
# u_real = [[u_analytic(x, t) for x in xs] for t in ts]
# diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1.estimated_nn_params[1]))))
#            for x in xs]
#           for t in ts]

# # p1 = plot(xs, u_predict, title = "predict")
# # p2 = plot(xs, u_real, title = "analytic")
# # p3 = plot(xs, diff_u, title = "error")
# # plot(p1, p2, p3)

# phi = discretization.phi[1]
# xs, ts = [infimum(d.domain):dx:supremum(d.domain)
#           for (d, dx) in zip(domains, [dx / 10, dt])]
# u_predict = [[first(pmean(phi([x, t], sol2.estimated_nn_params[1]))) for x in xs]
#              for t in ts]
# u_real = [[u_analytic(x, t) for x in xs] for t in ts]
# diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol2.estimated_nn_params[1]))))
#            for x in xs]
#           for t in ts]

# # p1 = plot(xs, u_predict, title = "predict")
# # p2 = plot(xs, u_real, title = "analytic")
# # p3 = plot(xs, diff_u, title = "error")
# # plot(p1, p2, p3)

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

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(4.0, 2)], progress = true)

param = 2 * π
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func1(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])

@test u_predict≈u_real atol=0.1
@test mean(u_predict .- u_real) < 0.01
@test sol1.estimated_de_params[1]≈param atol=0.1
sol1.estimated_de_params[1]

eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.05],
    phystd = [0.01], l2std = [0.02], phystdnew = [0.02],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    param = [LogNormal(4.0, 2)],
    Dict_differentials = Dict_differentials,
    progress = true)

param = 2 * π
ts_2 = vec(sol2.timepoints[1])
u_real_2 = [analytic_sol_func1(0.0, t) for t in ts]
u_predict_2 = pmean(sol2.ensemblesol[1])

@test u_predict_2≈u_real_2 atol=0.1
@test mean(u_predict_2 .- u_real_2) < 0.01
@test sol2.estimated_de_params[1]≈param atol=0.1
sol2.estimated_de_params[1]

plot(ts_2, u_predict_2)
plot!(ts_2, u_real_2)

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
    draw_samples = 100,
    bcstd = [0.3, 0.3, 0.3],
    phystd = [0.1, 0.1, 0.1],
    l2std = [1, 1, 1],
    priorsNNw = (0.0, 1.0),
    saveats = [0.01],
    param = [Normal(14.0, 2)], progress = true)

idealp = 10.0
p_ = sol1.estimated_de_params[1]
@test sum(abs, pmean(p_) - 10.00) < 0.3 * idealp[1]
# @test sum(abs, pmean(p_[2]) - (8 / 3)) < 0.3 * idealp[2]

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
discretization = BayesianPINN([chain], GridTraining(dx), dataset = [[dataset], nothing])

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 5,
    bcstd = [0.01, 0.01, 0.01, 0.01],
    phystd = [0.005],
    priorsNNw = (0.0, 2.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

xs = sol1.timepoints[1]
sol1.ensemblesol[1]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

dataset = hcat(u_real, xs')
u_predict = pmean(sol1.ensemblesol[1])
u_real = [analytic_sol_func(xs[:, i][1], xs[:, i][2]) for i in 1:length(xs[1, :])]
@test u_predict≈u_real atol=0.8

using NeuralPDE, Lux, ModelingToolkit, LinearAlgebra, AdvancedHMC
import ModelingToolkit: Interval, infimum, supremum, Distributions
using Plots, MonteCarloMeasurements

@parameters x t
@variables u(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Dx^2
α = 0.05
# Burger's equation
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - α * Dxx(u(t, x)) ~ 0

# boundary conditions
bcs = [
    u(0.0, x) ~ -sin(π * x),
    u(t, -1.0) ~ 0.0,
    u(t, 1.0) ~ 0.0
]

domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(-1.0, 1.0)]

# Neural network
chain = Lux.Chain(Dense(2, 10, Lux.σ), Dense(10, 10, Lux.σ), Dense(10, 1))
strategy = NeuralPDE.QuadratureTraining(; abstol = 1e-6, reltol = 1e-6, batch = 200)

indvars = [t, x]
depvars = [u(t, x)]
@named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

# KS EQUATION
using NeuralPDE, Flux, Lux, ModelingToolkit, LinearAlgebra, AdvancedHMC
import ModelingToolkit: Interval, infimum, supremum, Distributions
using Plots, MonteCarloMeasurements, StatsPlots

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

#   x_values = -10:dx:10
#     t_values = 0.0:dt:1.0

#     dataset = []

#     for t in t_values
#         for x in x_values
#             u_value = u_analytic_point(x, t)
#             push!(dataset, [u_value, x, t])
#         end
#     end
# dataset
#    pop= vcat([data' for data in dataset]...)

datasetpde = [generate_dataset_matrix(domains, dx, dt, [-10,10], [0.0,1.0])]

datasetpde_new = [generate_dataset_matrix(domains, dx, dt, [-10,0], [0.0,1.0])]

# noise to dataset
noisydataset = deepcopy(datasetpde)
noisydataset[1][:, 1] = noisydataset[1][:, 1] .+ (randn(size(noisydataset[1][:, 1])) .* 0.8)

noisydataset_new = deepcopy(datasetpde_new)
noisydataset_new[1][:, 1] = noisydataset_new[1][:, 1] .+ (randn(size(noisydataset_new[1][:, 1])) .* 0.8)

# a=discretization_new.dataset[1]

plotly()
plot(datasetpde[1][:, 2], datasetpde[1][:, 1], title = "Dataset from Analytical Solution")
# scatter!(a[1][:, 2], a[1][:, 1])
scatter!(noisydataset[1][:, 2], noisydataset[1][:, 1])

plot(datasetpde[1][:, 2],datasetpde[1][:, 3], datasetpde[1][:, 1], title = "Dataset from Analytical Solution")
scatter!(noisydataset[1][:, 2],noisydataset[1][:, 3], noisydataset[1][:, 1])

plotly()
plot(datasetpde_new[1][:, 2], datasetpde_new[1][:, 1], title = "Dataset from Analytical Solution")
scatter!(noisydataset_new[1][:, 2], noisydataset_new[1][:, 1])

plot(datasetpde_new[1][:, 2],datasetpde_new[1][:, 3], datasetpde_new[1][:, 1], title = "Dataset from Analytical Solution")
scatter!(noisydataset_new[1][:, 2],noisydataset_new[1][:, 3], noisydataset_new[1][:, 1])

noise_std = 1.4
original_data = datasetpde[1][:, 1]
original_std = std(original_data)
ratio = noise_std / original_std


using StatsPlots
plot(sol1.original.mcmc_chain)
plot(sol2.original.mcmc_chain)

plot(sol0_new.original.mcmc_chain)
plot(sol2_new.original.mcmc_chain)

# Neural network
chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh),
    Lux.Dense(8, 8, Lux.tanh),
    Lux.Dense(8, 1))

chain_more = Lux.Chain(Lux.Dense(2, 10, Lux.tanh),
    Lux.Dense(10, 10, Lux.tanh),
    Lux.Dense(10, 1))
# chain = Lux.Chain(Lux.Dense(2, 8, Lux.σ),
#     Lux.Dense(8, 8, Lux.σ),
#     Lux.Dense(8, 1))

discretization = NeuralPDE.BayesianPINN([chain],
    GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset, nothing])

discretization_more = NeuralPDE.BayesianPINN([chain_more],
    GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset, nothing])

discretization_new = NeuralPDE.BayesianPINN([chain],
    GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset_new, nothing])


@named pde_system = PDESystem(eq,
    bcs,
    domains,
    [x, t],
    [u(x, t)],
    [α],
    defaults = Dict([α => 2.0]))

eqs = pde_system.eqs
Dict_differentials = Dict()
exps = toexpr.(eqs)
nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

Dict_differentials

plot(sol1.original.mcmc_chain)
meanplot(sol1.original.mcmc_chain)
autocorplot(sol1.original.mcmc_chain)
traceplot(sol1.original.mcmc_chain)

plot(sol2.original.mcmc_chain)
meanplot(sol2.original.mcmc_chain)
autocorplot(sol2.original.mcmc_chain)
traceplot(sol2.original.mcmc_chain)

plot(sol0_new.original.mcmc_chain)
meanplot(sol0_new.original.mcmc_chain)
autocorplot(sol0_new.original.mcmc_chain)

plot(sol2_new.original.mcmc_chain)
meanplot(sol2_new.original.mcmc_chain)
autocorplot(sol2_new.original.mcmc_chain)

plot(sol3_new.original.mcmc_chain)
meanplot(sol3_new.original.mcmc_chain)
autocorplot(sol3_new.original.mcmc_chain)


sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 80, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.5],
    phystd = [0.5], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol1_more = ahmc_bayesian_pinn_pde(pde_system,
    discretization_more;
    draw_samples = 80, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.5],
    phystd = [0.5], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol1_1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 90, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew =  [0.7],
    phystd = [0.7], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 80, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.5], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

sol2_more = ahmc_bayesian_pinn_pde(pde_system,
    discretization_more;
    draw_samples = 80, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.5], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

# julia> sol2 = ahmc_bayesian_pinn_pde(pde_system,
#            discretization;
#            draw_samples = 85, Kernel = AdvancedHMC.NUTS(0.8),
#            bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
#            phystd = [0.7], l2std = [0.2], param = [Distributions.Normal(2.0,
#            priorsNNw = (0.0, 3.0),
#            saveats = [1 / 100.0, 1 / 100.0],
#            progress = true)
# ┌ Info: Current Physics Log-likelihood : 
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, initial_θ), ℓπ.allstd) = -415167
# ┌ Info: Current Prior Log-likelihood : 
# └   priorlogpdf(ℓπ, initial_θ) = -214.1825373360679
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, initial_θ) = -109309.44973223892
# Sampling 100%|███████████████████████████████| Time: 0:14:50
#   iterations:                                   85
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.02
#   n_steps:                                      127
#   is_accept:                                    true
#   acceptance_rate:                              0.9982795867682919
#   log_density:                                  -3832.934953640867
#   hamiltonian_energy:                           4145.005901868316
#   hamiltonian_energy_error:                     -0.07863051782624098
#   max_hamiltonian_energy_error:                 -0.16790754244266282
#   tree_depth:                                   7
#   numerical_error:                              false
#   step_size:                                    0.00018186972987192408
#   nom_step_size:                                0.00018186972987192408
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.
# [ Info: Sampling Complete.
# ┌ Info: Current Physics Log-likelihood :
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, samples[end]), ℓπ.allstd) = -132
# ┌ Info: Current Prior Log-likelihood :
# └   priorlogpdf(ℓπ, samples[end]) = -219.17544656823006
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, samples[end]) = -3481.509412470054

# julia> sol1 = ahmc_bayesian_pinn_pde(pde_system,
#            discretization;
#            draw_samples = 90, Kernel = AdvancedHMC.NUTS(0.8),
#            bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.7],
#            phystd = [0.7], l2std = [0.2], param = [Distributions.Normal(2.0,
#            priorsNNw = (0.0, 3.0),
#            saveats = [1 / 100.0, 1 / 100.0],
#            Dict_differentials = Dict_differentials,
#            progress = true)
# ┌ Info: Current Physics Log-likelihood : 
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, initial_θ), ℓπ.allstd) = -394622
# ┌ Info: Current Prior Log-likelihood : 
# └   priorlogpdf(ℓπ, initial_θ) = -214.1657203956881
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, initial_θ) = -107600.2750860966
# ┌ Info: Current L2_LOSSY : 
# └   ℓπ.L2_loss2(setparameters(ℓπ, initial_θ), ℓπ.allstd) = -352.339686469935
# Sampling 100%|███████████████████████████████| Time: 0:38:42
#   iterations:                                   90
#   ratio_divergent_transitions:                  0.24
#   ratio_divergent_transitions_during_adaption:  0.02
#   n_steps:                                      34
#   is_accept:                                    true
#   acceptance_rate:                              0.0755469536430885
#   log_density:                                  -6535.135018473582
#   hamiltonian_energy:                           6681.540376258076
#   hamiltonian_energy_error:                     -1.7097735125544204
#   max_hamiltonian_energy_error:                 1216.239238705054
#   tree_depth:                                   5
#   numerical_error:                              true
#   step_size:                                    0.0004111092751764056
#   nom_step_size:                                0.0004111092751764056
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.
# [ Info: Sampling Complete.
# ┌ Info: Current Physics Log-likelihood : 
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, samples[end]), ℓπ.allstd) = -272
# ┌ Info: Current Prior Log-likelihood : 
# └   priorlogpdf(ℓπ, samples[end]) = -218.6535874132563
# ┌ Info: Current MSE against dataset Log-likelihood : 
# └   L2LossData(ℓπ, samples[end]) = -3573.449092586736
# ┌ Info: Current L2_LOSSY : 
# └   ℓπ.L2_loss2(setparameters(ℓπ, samples[end]), ℓπ.allstd) = -2470.35523478

using MCMCChains
println(summarize(sol1.original.mcmc_chain))
plot(sol1.original.mcmc_chain)

sol3 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 100, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.7], l2std = [0.15], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 3.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

sol4 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 80, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.7], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)


phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol1.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol1_1.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1_1.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol2.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol2.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol3.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol3.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol4.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol4.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)

sol0_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 110, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.2],
    phystd = [0.2], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)


julia> sol5_new = ahmc_bayesian_pinn_pde(pde_system,
           discretization_new;
           draw_samples = 170, Kernel = AdvancedHMC.NUTS(0.8),
           bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
           phystd = [0.2], l2std = [0.1], param = [Distributions.Normal(2.0, 2)],
           priorsNNw = (0.0, 1.0),
           saveats = [1 / 100.0, 1 / 100.0],
           progress = true)

sol1_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 110, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.2],
    phystd = [0.5], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol1_1_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 110, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew =  [0.5],
    phystd = [0.5], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol1_2_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 150, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew =  [0.1],
    phystd = [0.2], l2std = [0.1], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 3.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol1_3_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 150, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew =  [0.2],
    phystd = [0.3], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 3.0),
    saveats = [1 / 100.0, 1 / 100.0],
    Dict_differentials = Dict_differentials,
    progress = true)

sol2_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 140, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.2], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)


sol3_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 140, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.2], l2std = [0.1], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

sol4_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 160, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.2], l2std = [0.1], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

sol5_new = ahmc_bayesian_pinn_pde(pde_system,
    discretization_new;
    draw_samples = 170, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.2], l2std = [0.1], param = [Distributions.Normal(2.0, 2)],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

# phi = discretization.phi[1]

phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol0_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol0_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)

phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol1_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol1_1_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1_1_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)

phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol1_2_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1_2_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol1_3_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1_3_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol2_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol2_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol3_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol3_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol4_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol4_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)


phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol5_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol5_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)

phi = discretization_new.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]

u_predict = [[first(pmean(phi([x, t], sol5_new.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol5_new.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]
plotly()
p1 = plot(ts, xs, u_predict, title = "predict")
p2 = plot(ts, xs, u_real, title = "analytic")
p3 = plot(ts, xs, diff_u, title = "error")
plot(p1, p2, p3)
# julia> sol0_new = ahmc_bayesian_pinn_pde(pde_system,
#            discretization_new;
#            draw_samples = 110, Kernel = AdvancedHMC.NUTS(0.8),
#            bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phystdnew = [0.2],
#            phystd = [0.2], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
#            priorsNNw = (0.0, 1.0),
#            saveats = [1 / 100.0, 1 / 100.0],
#            Dict_differentials = Dict_differentials,
#            progress = true)
# ┌ Info: Current Physics Log-likelihood :
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, initial_θ), ℓπ.allstd) = -398314.38213382766
# ┌ Info: Current Prior Log-likelihood :
# └   priorlogpdf(ℓπ, initial_θ) = -104.7365701596561
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, initial_θ) = -58553.36940699288
# ┌ Info: Current L2_LOSSY :
# └   ℓπ.L2_loss2(setparameters(ℓπ, initial_θ), ℓπ.allstd) = -281.85131447737575
# Sampling 100%|███████████████████████████████| Time: 0:26:00
#   iterations:                                   110
#   ratio_divergent_transitions:                  0.2
#   ratio_divergent_transitions_during_adaption:  0.03
#   n_steps:                                      11
#   is_accept:                                    true
#   acceptance_rate:                              0.0024891070448310416
#   log_density:                                  -13158.729119075539
#   hamiltonian_energy:                           13212.763613683248
#   hamiltonian_energy_error:                     0.0
#   max_hamiltonian_energy_error:                 1492.7356803165876
#   tree_depth:                                   3
#   numerical_error:                              true
#   step_size:                                    0.0002145156661425442
#   nom_step_size:                                0.0002145156661425442
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...])
# [ Info: Sampling Complete.
# ┌ Info: Current Physics Log-likelihood :
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, samples[end]), ℓπ.allstd) = -908.7769621441158
# ┌ Info: Current Prior Log-likelihood :
# └   priorlogpdf(ℓπ, samples[end]) = -136.87645881663929
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, samples[end]) = -1404.7102059521355
# ┌ Info: Current L2_LOSSY :
# └   ℓπ.L2_loss2(setparameters(ℓπ, samples[end]), ℓπ.allstd) = -10708.363203924739

# julia> sol2_new = ahmc_bayesian_pinn_pde(pde_system,
#            discretization_new;
#            draw_samples = 140, Kernel = AdvancedHMC.NUTS(0.8),
#            bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
#            phystd = [0.2], l2std = [0.2], param = [Distributions.Normal(2.0, 2)],
#            priorsNNw = (0.0, 1.0),
#            saveats = [1 / 100.0, 1 / 100.0],
#            progress = true)
# ┌ Info: Current Physics Log-likelihood :
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, initial_θ), ℓπ.allstd) = -397526.19267355377
# ┌ Info: Current Prior Log-likelihood :
# └   priorlogpdf(ℓπ, initial_θ) = -105.03439044100367
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, initial_θ) = -60957.24454333089
# Sampling  99%|███████████████████████████████|  ETA: 0:00:10
#   iterations:                                   140
#   ratio_divergent_transitions:                  0.0
#   ratio_divergent_transitions_during_adaption:  0.01
#   n_steps:                                      1023
#   is_accept:                                    true
#   acceptance_rate:                              0.972620625460237
#   log_density:                                  -1513.1769839294327
#   hamiltonian_energy:                           8709.204139640105
#   hamiltonian_energy_error:                     -0.4925547801958601
#   max_hamiltonian_energy_error:                 -1.7861646674082294
#   tree_depth:                                   10
#   numerical_error:                              false
#   step_size:                                    0.00011428277138492957
#   nom_step_size:                                0.00011428277138492957
#   is_adapt:                                     false
#   mass_matrix:                                  DiagEuclideanMetric([1.0, 1.0, 1.0, 1.0, 1.0, 1 ...])
# [ Info: Sampling Complete.
# ┌ Info: Current Physics Log-likelihood :
# └   ℓπ.full_loglikelihood(setparameters(ℓπ, samples[end]), ℓπ.allstd) = 115.103823132341
# ┌ Info: Current Prior Log-likelihood :
# └   priorlogpdf(ℓπ, samples[end]) = -198.39103020815858
# ┌ Info: Current MSE against dataset Log-likelihood :
# └   L2LossData(ℓπ, samples[end]) = -1429.7843027541815
