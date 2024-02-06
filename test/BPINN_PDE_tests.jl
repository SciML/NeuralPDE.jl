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

using NeuralPDE, Flux, Lux, ModelingToolkit, LinearAlgebra, AdvancedHMC
import ModelingToolkit: Interval, infimum, supremum, Distributions

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

using Plots, MonteCarloMeasurements, StatsPlots
plotly()

datasetpde = [generate_dataset_matrix(domains, dx, dt)]
plot(datasetpde[1][:, 2], datasetpde[1][:, 1])

# Add noise to dataset
datasetpde[1][:, 1] = datasetpde[1][:, 1] .+
                      randn(size(datasetpde[1][:, 1])) .* 5 / 100 .*
                      datasetpde[1][:, 1]
plot!(datasetpde[1][:, 2], datasetpde[1][:, 1])

# Neural network
chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh),
    Lux.Dense(8, 8, Lux.tanh),
    Lux.Dense(8, 1))

discretization = NeuralPDE.BayesianPINN([chain],
    adaptive_loss = GradientScaleAdaptiveLoss(5),
    # MiniMaxAdaptiveLoss(5),
    GridTraining([dx, dt]), param_estim = true, dataset = [datasetpde, nothing])
@named pde_system = PDESystem(eq,
    bcs,
    domains,
    [x, t],
    [u(x, t)],
    [α],
    defaults = Dict([α => 0.5]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 1000,
     Kernel = AdvancedHMC.NUTS(0.80),
    bcstd = [1.0, 1.0, 1.0, 1.0, 1.0],
    phystd = [0.1], l2std = [0.05], param = [Distributions.LogNormal(0.5, 2)],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0], progress = true)

phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, [dx / 10, dt])]
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
plot(datasetpde[1][:, 2], datasetpde[1][:, 1], title = "Dataset from Analytical Solution")

# Add noise to dataset
datasetpde[1][:, 1] = datasetpde[1][:, 1] .+
                      randn(size(datasetpde[1][:, 1])) .* 5 / 100 .*
                      datasetpde[1][:, 1]
plot!(datasetpde[1][:, 2], datasetpde[1][:, 1])

function CostFun(x::AbstractVector{T}) where {T}
    function SpringEqu!(du, u, x, t)
        du[1] = u[2]
        du[2] = -(x[1] / x[3]) * u[2] - (x[2] / x[3]) * u[1] + 50 / x[3]
    end

    u0 = T[2.0, 0.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(SpringEqu!, u0, tspan, x)
    sol = solve(prob)

    Simpos = zeros(T, length(sol.t))
    Simvel = zeros(T, length(sol.t))
    tout = zeros(T, length(sol.t))
    for i in 1:length(sol.t)
        tout[i] = sol.t[i]
        Simpos[i] = sol[1, i]
        Simvel[i] = sol[2, i]
    end

    totalCost = sum(Simpos)
    return totalCost
end

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

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
chain = Lux.Chain(Lux.Dense(2, 8, Lux.tanh),
    Lux.Dense(8, 8, Lux.tanh),
    Lux.Dense(8, 1))

discretization = PhysicsInformedNN(chain,
    adaptive_loss = GradientScaleAdaptiveLoss(1),
    GridTraining([dx, dt]))
@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

opt = OptimizationOptimJL.BFGS()
res = Optimization.solve(prob, opt; callback = callback, maxiters = 100)
phi = discretization.phi