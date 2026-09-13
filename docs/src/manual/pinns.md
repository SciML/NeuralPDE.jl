# `PhysicsInformedNN` Discretizer for PDESystems

Using the PINNs solver, we can solve general nonlinear PDEs:

```math
f{\left(x; \frac{∂u}{∂x_1}, \dots, \frac{∂u}{∂x_d}; \frac{∂^2 u}{∂x_1 ∂x_1}, \frac{∂^2 u}{∂x_1 ∂x_d}; \dots ; \lambda\right)} = 0, x \in \Omega,
```

with suitable boundary conditions:

```math
B(u, x) = 0 \; \text{ on } \; ∂\Omega
```

where time t is a special component of x, and Ω contains the temporal domain.

PDEs are defined using the ModelingToolkit.jl `PDESystem`:

```julia
@named pde_system = PDESystem(eq, bcs, domains, param, var)
```

Here, `eq` is the equation, `bcs` represents the boundary conditions, `param` is
the parameter of the equation (like `[x,y]`), and `var` represents variables (like `[u]`).
For more information, see the
[ModelingToolkit.jl PDESystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/API/PDESystem/).

## The pipeline

`PhysicsInformedNN` follows the same `PDESystem → System → problem → solution` pipeline
as the other SciML discretizers (for example MethodOfLines.jl):

1. `symbolic_discretize(pdesys, discretization)` returns a `ModelingToolkit.System`. Each
   dependent variable is represented by a `ModelingToolkitNeuralNets.SymbolicNeuralNetwork`
   whose parameters are array unknowns of the system; the residual of every equation and
   boundary condition is a symbolic array expression over a matrix of collocation points
   stored as a parameter of the system; each residual contributes one cost (or, for
   boundary conditions with `boundary_policy = :constraints`, one equality constraint).
2. `discretize(pdesys, discretization)` samples the collocation points with the training
   strategy and generates an `OptimizationProblem` through ModelingToolkit's
   `OptimizationProblem(sys, op)` constructor. Keyword arguments such as `adtype`
   (automatic differentiation backend) and `weights` (weighted sum of the costs) are
   forwarded to it. The system is only `complete`d, not passed through `mtkcompile`:
   the network parameters stay array unknowns of the problem, which keeps the generated
   code independent of the number of parameters.
3. `solve(prob, optimizer)` returns a `PDENoTimeSolution` which evaluates the trained
   networks: `sol[u(x, t)]` on the evaluation grid, `sol(x, t; dv = u(x, t))` at arbitrary
   points, and `sol.original_sol` for the underlying `OptimizationSolution`.

## The `PhysicsInformedNN` Discretizer

```@docs
NeuralPDE.PhysicsInformedNN
NeuralPDE.FiniteDifferenceDerivative
SciMLBase.discretize(::PDESystem, ::NeuralPDE.PhysicsInformedNN)
NeuralPDE.default_adtype
```

## `symbolic_discretize` and the generated `System`

```@docs
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.PhysicsInformedNN)
NeuralPDE.pinn_metadata
NeuralPDE.PINNMetadata
NeuralPDE.TrialNetwork
NeuralPDE.ResidualBlock
NeuralPDE.nn_eval
NeuralPDE.nn_eval_row
```

## Solutions, resampling and transfer learning

```@docs
SciMLBase.PDENoTimeSolution(::SciMLBase.AbstractOptimizationSolution, ::NeuralPDE.PINNMetadata)
NeuralPDE.trial_function
NeuralPDE.resample!
```

Warm-starting a problem from previously trained weights (transfer learning) is
`remake(prob; u0 = trained_weights)`, and replacing the collocation points of a residual
block is `remake(prob; p = [block.xs => new_points])`.

## SDE Solvers

```@docs
NeuralPDE.NNSDE
NeuralPDE.SDEPINN
```
