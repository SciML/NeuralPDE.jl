# Debugging PINN Solutions

The physics-informed loss is generated symbolically, so every piece of it can be
inspected before and after training.

## Helper API

```@docs
NeuralPDE.get_loss_function
NeuralPDE.vector_to_parameters
```

## Inspecting the generated `System`

`symbolic_discretize` returns the `ModelingToolkit.System` whose costs are the mean
squared residuals of the PDEs and boundary conditions. The
[`NeuralPDE.PINNMetadata`](@ref) attached to it lists every residual block with its
free independent variables, collocation array and lowered residual expression:

```@example debugging
using ModelingToolkit, NeuralPDE, Lux
import DomainSets: Interval

@parameters x t
@variables u(..)
Dxx = Differential(x)^2
Dt = Differential(t)
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

chain = Chain(Dense(2, 8, tanh), Dense(8, 1))
discretization = PhysicsInformedNN(chain, GridTraining(0.1))
sys = symbolic_discretize(pde_system, discretization)
```

```@example debugging
md = pinn_metadata(sys)
[(b.kind, b.ivs, b.npoints) for b in md.blocks]
```

```@example debugging
md.blocks[1].residual
```

The costs of the system are the scalar reductions of these residuals:

```@example debugging
ModelingToolkit.get_costs(sys)
```

## Evaluating individual residuals and costs

`discretize` compiles the system into an `OptimizationProblem`. The residual of any
block, or any other symbolic expression of the unknowns and parameters, can be evaluated
on a problem or solution through SymbolicIndexingInterface:

```@example debugging
using SymbolicIndexingInterface: getu, getp
prob = discretize(pde_system, discretization)
md = pinn_metadata(prob)
residual_of_pde = getu(prob, md.blocks[1].residual)
size(residual_of_pde(prob))
```

```@example debugging
collocation_points = getp(prob, md.blocks[1].xs)(prob)
size(collocation_points)
```

The objective itself is `prob.f(prob.u0, prob.p)`, and its gradient can be checked with
any AD backend, for example `ForwardDiff.gradient(u -> prob.f(u, prob.p), prob.u0)`.
Passing `weights` to `discretize` scales the individual costs, which is a quick way to
see which residual dominates the objective.
