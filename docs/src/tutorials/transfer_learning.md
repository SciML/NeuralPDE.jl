# Transfer Learning by Distillation

A trained PINN can be reused in two ways. Restarting training in the *same*
architecture from known weights is a warm start with `remake(prob; u0 = θ)`. Moving
the learned solution into a *different* architecture is distillation: fit the new
network to the evaluations of the trained solution with [`distill`](@ref). Both are
shown below on the 2D Poisson equation

```math
\partial^2_x u + \partial^2_y u = -\sin(\pi x)\sin(\pi y)
```

with zero Dirichlet boundary conditions on ``[0, 1]^2``, whose analytic solution is
``u(x, y) = \sin(\pi x)\sin(\pi y) / (2\pi^2)``.

We first train a small teacher network:

```@example transfer
using NeuralPDE, Lux, OptimizationOptimJL, LineSearches, Random
using DomainSets: Interval

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))
disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(1))
prob = discretize(pde_system, disc)
teacher = solve(prob, LBFGS(linesearch = BackTracking()); maxiters = 500)
teacher.original_sol.objective
```

## Same-architecture warm starts

`remake(prob; u0 = θ)` rebuilds the training problem starting from `θ`, for example
the weights of a previous solve, a related PDE, or a partially trained run:

```@example transfer
warm_prob = remake(prob; u0 = teacher.original_sol.u)
refined = solve(warm_prob, LBFGS(linesearch = BackTracking()); maxiters = 100)
refined.original_sol.objective
```

## Distilling into a different architecture

`distill(teacher, student; ...)` builds an `OptimizationProblem` whose objective is
the mean squared error of the student network against `teacher(x, y; dv = u(x, y))`
on a batch of points (drawn uniformly with `rng`, or passed explicitly as `points`).
Solving it transfers the solution into the new architecture without touching the PDE
residuals:

```@example transfer
student = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
dprob = distill(teacher, student; npoints = 1000, rng = Xoshiro(2))
dres = solve(dprob, LBFGS(linesearch = BackTracking()); maxiters = 300)
dres.original_sol.objective
```

The distilled weights are the flat vector `dres.u`. Wrapping them back into the
network's parameter container evaluates the student anywhere:

```@example transfer
using ComponentArrays: ComponentArray, getaxes
template, st = Lux.setup(Xoshiro(0), student)
θ = ComponentArray(dres.u, getaxes(ComponentArray(template)))
u_student(x, y) = only(first(student(reshape(Float64[x, y], 2, 1), θ, st)))
analytic(x, y) = sinpi(x) * sinpi(y) / (2pi^2)
maximum(abs, [u_student(x, y) - analytic(x, y) for x in 0:0.05:1, y in 0:0.05:1])
```

```@example transfer
maximum(abs, [u_student(x, y) - teacher(x, y; dv = u(x, y)) for x in 0:0.05:1, y in 0:0.05:1])
```

## Distilling a domain decomposition

A decomposition trains one small network per subdomain and distills them all into a
single global network. Each subdomain is an ordinary `PDESystem` with the analytic
solution (or the neighboring network's prediction) as its interface data.
`distill` accepts the vector of subdomain solutions and evaluates each teacher on
its own points:

```@example transfer
left_domains = [x ∈ Interval(0.0, 0.5), y ∈ Interval(0.0, 1.0)]
right_domains = [x ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0)]
left_bcs = [u(0, y) ~ 0.0, u(0.5, y) ~ analytic(0.5, y), u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
right_bcs = [u(0.5, y) ~ analytic(0.5, y), u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
@named left_system = PDESystem(eq, left_bcs, left_domains, [x, y], [u(x, y)])
@named right_system = PDESystem(eq, right_bcs, right_domains, [x, y], [u(x, y)])
subchain() = Chain(Dense(2, 8, σ), Dense(8, 8, σ), Dense(8, 1))
left_sol = solve(
    discretize(left_system, PhysicsInformedNN(subchain(), GridTraining(0.2); rng = Xoshiro(3))),
    LBFGS(linesearch = BackTracking()); maxiters = 200)
right_sol = solve(
    discretize(right_system, PhysicsInformedNN(subchain(), GridTraining(0.2); rng = Xoshiro(4))),
    LBFGS(linesearch = BackTracking()); maxiters = 200)
global_net = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
gres = solve(
    distill([left_sol, right_sol], global_net; npoints = 500, rng = Xoshiro(5)),
    LBFGS(linesearch = BackTracking()); maxiters = 200)
gtemplate, gst = Lux.setup(Xoshiro(0), global_net)
gθ = ComponentArray(gres.u, getaxes(ComponentArray(gtemplate)))
u_global(x, y) = only(first(global_net(reshape(Float64[x, y], 2, 1), gθ, gst)))
maximum(abs, [u_global(x, y) - analytic(x, y) for x in 0:0.05:1, y in 0:0.05:1])
```
