using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays

Random.seed!(100)

# Cospit example
@parameters t
@variables u(..)

Dt = Differential(t)

eqs = Dt(u(t)) - cos(2 * π * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1)) |> Flux.f64
init1, re1 = Flux.destructure(chainf)
chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

# non adaptive case
discretization = NeuralPDE.BayesianPINN([chainl], GridTraining([0.01]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.02],
    phystd = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0])

discretization = NeuralPDE.BayesianPINN([chainf], GridTraining([0.01]))
sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.01],
    phystd = [0.005],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0])

analytic_sol_func(u0, t) = u0 + sin(2 * π * t) / (2 * π)
ts = vec(sol1.timepoints[1])
u_real = [analytic_sol_func(0.0, t) for t in ts]
u_predict = pmean(sol1.ensemblesol[1])
@test u_predict≈u_real atol=0.5
@test mean(u_predict .- u_real) < 0.1

ts = vec(sol2.timepoints[1])
u_real = [analytic_sol_func(0.0, t) for t in ts]
u_predict = pmean(sol2.ensemblesol[1])
@test u_predict≈u_real atol=0.5
@test mean(u_predict .- u_real) < 0.1

## Example 1, 1D ode
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
chain = Lux.Chain(Lux.Dense(1, 12, Flux.σ), Lux.Dense(12, 1))

discretization = NeuralPDE.BayesianPINN([chain],
    GridTraining([0.01]))

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

# example 3 (3 degree ODE)
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

discretization = NeuralPDE.BayesianPINN(chain, GridTraining(0.01))

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

# diff_u = abs.(u_real .- u_predict)
# plot(xs, u_real)
# plot!(xs, u_predict)
# plot!(xs, diff_u)

# 2D Poissons equation 
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
dx = 0.05
discretization = NeuralPDE.BayesianPINN([chain], GridTraining(dx))

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
diff_u = abs.(u_predict .- u_real)
@test u_predict≈u_real atol=1.5

# using Plots, StatsPlots
# plotly()
# plot(sol1.timepoints[1][1, :],
#     sol1.timepoints[1][2, :],
#     pmean(sol1.ensemblesol[1]),
#     linetype = :contourf)
# plot(sol1.timepoints[1][1, :], sol1.timepoints[1][2, :], u_real, linetype = :contourf)
# plot(sol1.timepoints[1][1, :], sol1.timepoints[1][2, :], diff_u, linetype = :contourf)

using NeuralPDE, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

# Initial and boundary conditions
bcs = [u(0, x) ~ -sin(pi * x),
    u(t, -1) ~ 0.0,
    u(t, 1) ~ 0.0,
    u(t, -1) ~ u(t, 1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(-1.0, 1.0)]
# Discretization
dx = 0.05
# Neural network
chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh), Lux.Dense(8, 8, Lux.tanh), Lux.Dense(8, 1))
strategy = NeuralPDE.GridTraining([dx, dx])

discretization = NeuralPDE.BayesianPINN([chain], strategy)

@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 200,
    bcstd = [0.01, 0.01, 0.01, 0.01],
    phystd = [0.01],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0], progress = true)

using Plots, StatsPlots
plotly()
plot(sol1.timepoints[1][2, :], sol1.timepoints[1][1, :],
    pmean(sol1.ensemblesol[1]),
    linetype = :contourf)

plot(sol1.timepoints[1][1, :], pmean(sol1.ensemblesol[1]))