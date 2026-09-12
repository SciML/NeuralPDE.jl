# Investigating `symbolic_discretize` with the `PhysicsInformedNN` Discretizer for the 1-D Burgers' Equation

Let's consider the Burgers' equation:

```math
\begin{gather*}
∂_t u + u ∂_x u - (0.01 / \pi) ∂_x^2 u = 0 \, , \quad x \in [-1, 1], t \in [0, 1] \, , \\
u(0, x) = - \sin(\pi x) \, , \\
u(t, -1) = u(t, 1) = 0 \, ,
\end{gather*}
```

with Physics-Informed Neural Networks. Here is an example of using the low-level API:

```@example low_level
using NeuralPDE, Lux, OptimizationOptimJL, LineSearches
using DomainSets: Interval

@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

# Initial and boundary conditions
bcs = [u(0, x) ~ -sinpi(x),
    u(t, -1) ~ 0.0,
    u(t, 1) ~ 0.0,
    u(t, -1) ~ u(t, 1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(-1.0, 1.0)]

# Neural network
chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
strategy = QuasiRandomTraining(500; bcs_points = 100)

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
discretization = PhysicsInformedNN(chain, strategy)
sys = symbolic_discretize(pde_system, discretization)
```

`symbolic_discretize` returns a `ModelingToolkit.System`. Its unknowns are the network
parameters, its parameters are the network callable and the collocation matrices, and
its costs are the mean squared residuals of the PDE and of each boundary condition:

```@example low_level
unknowns(sys)
```

```@example low_level
ModelingToolkit.parameters(sys)
```

```@example low_level
ModelingToolkit.get_costs(sys)
```

The periodic condition `u(t, -1) ~ u(t, 1)` is lowered like any other equation: both
calls of `u` are network evaluations on the same batch of `t` values with the pinned
`x` coordinate appended, so periodicity does not need special treatment:

```@example low_level
md = pinn_metadata(sys)
md.blocks[end].residual
```

Because the result is a `System`, the standard ModelingToolkit compiler builds the
`OptimizationProblem`. `discretize` does exactly this: it calls `mtkcompile`, samples the
collocation points and calls `OptimizationProblem(sys, op)`, forwarding keyword arguments
such as `adtype` or `weights`. Weighting the boundary conditions ten times more than the
PDE residual, for instance, is

```@example low_level
prob = discretize(pde_system, discretization; weights = [1.0, 10.0, 10.0, 10.0, 10.0])
sol = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 3000)
sol.original_sol.objective
```

And some analysis:

```@example low_level
using Plots

ts = 0:0.01:1
xs = -1:0.01:1
u_predict = sol(ts, xs; dv = u(t, x))
plot(ts, xs, u_predict', linetype = :contourf, title = "predict")

p1 = plot(xs, u_predict[3, :], title = "t = 0.02");
p2 = plot(xs, u_predict[51, :], title = "t = 0.5");
p3 = plot(xs, u_predict[end, :], title = "t = 1");
plot(p1, p2, p3)
```
