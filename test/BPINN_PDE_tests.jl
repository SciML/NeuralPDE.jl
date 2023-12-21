using Test, MCMCChains, Lux, ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using ForwardDiff, Distributions, OrdinaryDiffEq
using Flux, AdvancedHMC, Statistics, Random, Functors
using NeuralPDE, MonteCarloMeasurements
using ComponentArrays

# Forward solving example -1 
@parameters t
@variables u(..)

Dt = Differential(t)
linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
linear = (u, p, t) -> cos(2 * π * t)

eqs = Dt(u(t)) - cos(2 * π * t) ~ 0
bcs = [u(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 4.0)]

chainf = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 1))
init1, re1 = Flux.destructure(chainf)
chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 1))
initl, st = Lux.setup(Random.default_rng(), chainl)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u(t)])

# non adaptive case
discretization = NeuralPDE.PhysicsInformedNN([chainl], GridTraining([0.01]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1500,
    bcstd = [0.02],
    phystd = [0.01],
    priorsNNw = (0.0, 1.0),
    saveats = [1 / 50.0],
    progress = true)

discretization = NeuralPDE.PhysicsInformedNN([chainf], GridTraining([0.01]))
sol2 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 100,
    bcstd = [0.02],
    phystd = [0.01],
    priorsNNw = (0.0, 1.0),
    progress = true)

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

discretization = NeuralPDE.PhysicsInformedNN([chain],
    GridTraining([0.01]))

@named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.1],
    phystd = [0.05],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0],
    progress = true)

analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
u_real = [analytic_sol_func(t) for t in ts]
u_predict = sol1.ensemblesol[1]
@test pmean(u_predict)≈u_real atol=0.2

# example 3 (3 degree ODE)
@parameters x
@variables u(..), Dxu(..), Dxxu(..), O1(..), O2(..)
Dxxx = Differential(x)^3
Dx = Differential(x)

# ODE
eq = Dx(Dxxu(x)) ~ cos(pi * x)

# Initial and boundary conditions
bcs_ = [u(0.0) ~ 0.0,
    u(1.0) ~ cos(pi),
    Dxu(1.0) ~ 1.0]
ep = (cbrt(eps(eltype(Float64))))^2 / 6

der = [Dxu(x) ~ Dx(u(x)) + ep * O1(x),
    Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x)]

bcs = [bcs_; der]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0)]

# Neural network
chain = [
    Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
        Lux.Dense(12, 1)), Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
        Lux.Dense(12, 1)), Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
        Lux.Dense(12, 1)), Lux.Chain(Lux.Dense(1, 4, Lux.tanh), Lux.Dense(4, 1)),
    Lux.Chain(Lux.Dense(1, 4, Lux.tanh), Lux.Dense(4, 1))]

discretization = NeuralPDE.PhysicsInformedNN(chain, GridTraining(0.01))

@named pde_system = PDESystem(eq, bcs, domains, [x],
    [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.1, 0.1, 0.1],
    phystd = [0.05],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0],
    progress = true)

@parameters x
@variables u(..)

Dxxx = Differential(x)^3
Dx = Differential(x)
# ODE
eq = Dxxx(u(x)) ~ cos(pi * x)

# Initial and boundary conditions
bcs = [u(0.0) ~ 0.0,
    u(1.0) ~ cos(pi),
    Dx(u(1.0)) ~ 1.0]

# Space and time domains
domains = [x ∈ Interval(0.0, 1.0)]

analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
u_real = [analytic_sol_func(x) for x in xs]
plot
u_predict = [first(phi(x, res.u.depvar.u)) for x in xs]

@test u_predict≈u_real atol=10^-4

# Neural network
chain = Lux.Chain(Lux.Dense(1, 8, Lux.σ), Lux.Dense(8, 1))

discretization = PhysicsInformedNN([chain], GridTraining(0.01))
@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
    bcstd = [0.1, 0.1, 0.1],
    phystd = [0.05],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0],
    progress = true)

# KS equation
@parameters x, t
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dx2 = Differential(x)^2
Dx3 = Differential(x)^3
Dx4 = Differential(x)^4

α = 1
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

# Neural network
chain = [Lux.Chain(Lux.Dense(2, 8, Lux.σ), Lux.Dense(8, 8, Lux.σ), Lux.Dense(8, 1))]

discretization = PhysicsInformedNN(chain, GridTraining([dx, dt]))
@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 500,
    bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
    phystd = [0.05],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

#note that is KS equation and 3degree ode example std setting hasnt been done yet

# Poisson equation 
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
chain = Lux.Chain(Lux.Dense(dim, 10, Lux.σ), Lux.Dense(10, 10, Lux.σ), Lux.Dense(10, 1))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN([chain], GridTraining(dx))

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 200,
    bcstd = [0.001, 0.001, 0.001, 0.001],
    phystd = [0.001],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0],
    progress = true)

xs = sol1.timepoints[1]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)
u_predict = pmean(sol1.ensemblesol[1]) 
u_real = [analytic_sol_func(xs[:, i][1], xs[:, i][2]) for i in 1:length(xs[1, :])]

diff_u = abs.(u_predict .- u_real)
@test mean(diff_u)<0.1 
# @test u_predict≈u_real atol=2.0

# plotly()
# plot(sol1.timepoints[1][1, :],
#     sol1.timepoints[1][2, :],
#     pmean(sol1.ensemblesol[1]),
#     linetype = :contourf)


# plot(sol1.timepoints[1][1, :], sol1.timepoints[1][2, :], u_real, linetype = :contourf)
# plotly()
# plot(sol1.timepoints[1][1, :], sol1.timepoints[1][2, :], diff_u, linetype = :contourf)

# using Plots, StatsPlots
# plot(sol1.ensemblesol[1])
# sol1.ensemblesol[1]
# sol1.estimated_de_params
# sol1.estimated_nn_params
# sol1.original
# const T = MonteCarloMeasurements.Particles{Float64}
# NP <: Vector{Union{Vector{T}, T}},
# OP <: Union{Vector{Nothing}, Vector{T}, T},