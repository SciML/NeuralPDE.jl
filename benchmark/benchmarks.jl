using BenchmarkTools
using DomainSets: Interval, infimum, supremum
using Lux
using ModelingToolkit
using NeuralPDE
using Optimisers
using OrdinaryDiffEq
using Random
using SciMLBase

const SUITE = BenchmarkGroup()

Random.seed!(100)

linear = (u, p, t) -> @. t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
    u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))
linear_analytic = (u0, p, t) -> [exp(-(t^2) / 2) / (1 + t + t^3) + t^2]
ode_prob = ODEProblem(
    ODEFunction(linear; analytic = linear_analytic), [1.0f0], (0.0f0, 1.0f0)
)

SUITE["nnode"] = BenchmarkGroup()
SUITE["nnode"]["construct"] = @benchmarkable NNODE(
    Chain(Dense(1, 32, σ), Dense(32, 1)), Adam(0.01)
)
nnode_alg = NNODE(Chain(Dense(1, 32, σ), Dense(32, 1)), Adam(0.01))
SUITE["nnode"]["solve_50iters"] = @benchmarkable solve(
    $ode_prob, $nnode_alg; verbose = false, maxiters = 50, abstol = 1.0e-6
)

SUITE["pinn_poisson"] = BenchmarkGroup()

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

SUITE["pinn_poisson"]["pdesystem"] = @benchmarkable PDESystem(
    $eq, $bcs, $domains, [$x, $y], [$u($x, $y)]; name = :poisson
)

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
pinn_chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
strategy = StochasticTraining(64; bcs_points = 16)
ps = Lux.initialparameters(Random.default_rng(), pinn_chain)

SUITE["pinn_poisson"]["pinn_construct"] = @benchmarkable PhysicsInformedNN(
    $pinn_chain, $strategy
)
discretization = PhysicsInformedNN(pinn_chain, strategy; init_params = ps)
SUITE["pinn_poisson"]["discretize"] = @benchmarkable discretize(
    $pde_system, $discretization
)
prob_pinn = discretize(pde_system, discretization)
SUITE["pinn_poisson"]["solve_20iters"] = @benchmarkable solve(
    $prob_pinn, Adam(0.01); maxiters = 20, verbose = false
)
