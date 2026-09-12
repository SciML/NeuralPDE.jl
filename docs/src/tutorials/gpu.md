# Using GPUs to train Physics-Informed Neural Networks (PINNs)

!!! warning "GPU support for `PhysicsInformedNN` in NeuralPDE 7"

    The `PhysicsInformedNN` discretization of NeuralPDE 7 evaluates the networks through
    the `ModelingToolkitNeuralNets` symbolic network wrapper, whose parameter
    reconstruction is not yet device generic. GPU training of `PDESystem`s is therefore
    not supported in this release; it will return with an upstream
    `ModelingToolkitNeuralNets` change. Progress is tracked in
    https://github.com/SciML/NeuralPDE.jl/issues/1161. The ODE solvers (`NNODE`) keep
    their GPU support, see the [ODE tutorial](@ref).

The intended workflow is unchanged from the CPU case: the network parameters, the
collocation matrices and the trained weights are the arrays of the generated
`OptimizationProblem`, so moving them to the GPU with `remake` moves the whole
computation:

```julia
using NeuralPDE, Lux, LuxCUDA, OptimizationOptimisers
using DomainSets: Interval
const gpud = gpu_device()

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)

eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
bcs = [u(0, x, y) ~ analytic_sol_func(0, x, y),
    u(t, 0, y) ~ analytic_sol_func(t, 0, y),
    u(t, 2, y) ~ analytic_sol_func(t, 2, y),
    u(t, x, 0) ~ analytic_sol_func(t, x, 0),
    u(t, x, 2) ~ analytic_sol_func(t, x, 2)]
domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 2.0), y ∈ Interval(0.0, 2.0)]

inner = 25
chain = Chain(Dense(3, inner, σ), Dense(inner, inner, σ), Dense(inner, inner, σ),
    Dense(inner, inner, σ), Dense(inner, 1))
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(2000; bcs_points = 500))
@named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
prob = discretize(pde_system, discretization)

# Move the weights and every collocation matrix to the GPU.
md = pinn_metadata(prob)
gpu_points = [block.xs => gpud(getp(prob, block.xs)(prob)) for block in md.blocks]
prob = remake(prob; u0 = gpud(prob.u0), p = gpu_points)
sol = solve(prob, Adam(1e-2); maxiters = 2500)
```
