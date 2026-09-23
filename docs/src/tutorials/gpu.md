# Using GPUs to train Physics-Informed Neural Networks (PINNs)

`PhysicsInformedNN` supports GPU training when used with a device-compatible Lux network.
Move the initial network parameters and every collocation matrix and quadrature weight
with `remake` before solving. The returned `PDENoTimeSolution` evaluates on the host;
`sol.original_sol` retains the optimizer's device-resident result.

```julia
using NeuralPDE, Lux, LuxCUDA, OptimizationOptimisers
using DomainSets: Interval
using SymbolicIndexingInterface: getp
const gpud = gpu_device()

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)

eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
bcs = [
    u(0, x, y) ~ analytic_sol_func(0, x, y),
    u(t, 0, y) ~ analytic_sol_func(t, 0, y),
    u(t, 2, y) ~ analytic_sol_func(t, 2, y),
    u(t, x, 0) ~ analytic_sol_func(t, x, 0),
    u(t, x, 2) ~ analytic_sol_func(t, x, 2),
]
domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 2.0), y ∈ Interval(0.0, 2.0)]

inner = 25
chain = Chain(
    Dense(3, inner, σ), Dense(inner, inner, σ), Dense(inner, inner, σ),
    Dense(inner, inner, σ), Dense(inner, 1)
)
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(2000; bcs_points = 500))
@named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
prob = discretize(pde_system, discretization)

# Move the weights and every collocation matrix to the GPU.
md = pinn_metadata(prob)
gpu_parameters = [
    block.xs => gpud(getp(prob, block.xs)(prob)) for block in md.blocks if block.xs !== nothing
]
append!(
    gpu_parameters, [
        block.w => gpud(getp(prob, block.w)(prob)) for block in md.blocks if block.w !== nothing
    ]
)
prob = remake(prob; u0 = gpud(prob.u0), p = gpu_parameters)
sol = solve(prob, Adam(1.0e-2); maxiters = 2500)
```
