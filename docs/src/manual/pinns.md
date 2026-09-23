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
   boundary conditions with `boundary_policy = :constraints`, a pointwise equality).
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

## Exact boundary constraints

By default, boundary residuals contribute mean squared penalty costs. Set
`boundary_policy = :constraints` to impose each boundary residual as an equality at its
collocation points. The selected optimization solver must support equality constraints;
for example, `IpoptOptimizer` from OptimizationIpopt does. Constraint derivatives for
these array residuals need an automatic differentiation backend, so pass `adtype` when
discretizing:

```julia
using OptimizationIpopt
using ADTypes: AutoForwardDiff

disc = PhysicsInformedNN(chain, GridTraining(0.1); boundary_policy = :constraints)
prob = discretize(pde_system, disc; adtype = AutoForwardDiff())
sol = solve(prob, IpoptOptimizer(); maxiters = 1000)
```

Install OptimizationIpopt in the environment when using this solver; it brings the Ipopt
binary, which is distributed under the Eclipse Public License 2.0.

Started from the default initial weights, a constrained solve can converge to a trivial
feasible network: under homogeneous Dirichlet conditions `u ≡ 0` satisfies every boundary
equality exactly, so it is a stationary point of the constrained problem regardless of the
interior residual. Warm-start the constrained problem from a converged
`boundary_policy = :penalty` solve — a partial warm start can still collapse — and check
that the trained network is not the trivial boundary-satisfying function:

```julia
penalty_prob = discretize(pde_system,
    PhysicsInformedNN(chain, GridTraining(0.1); boundary_policy = :penalty);
    adtype = AutoForwardDiff())
warm = solve(penalty_prob, IpoptOptimizer(); maxiters = 1000)
sol = solve(remake(prob; u0 = warm.original_sol.u), IpoptOptimizer(); maxiters = 1000)
```

## Integral terms

`Symbolics.Integral` terms are supported: they lower to a fixed-node quadrature over
the batch of collocation points. For

```julia
@parameters t τ
@variables u(..)
I = Integral(τ in DomainSets.ClosedInterval(0.0, t))
@named sys = PDESystem([I(u(τ)) ~ t^2 / 2], [u(0.0) ~ 0.0],
    [t ∈ Interval(0.0, 1.0)], [t], [u(t)])
```

the integrating variable `τ` is declared with `@parameters` like any other symbol and
needs no domain of its own; the bounds may depend on the free independent variables
(`0.0 .. t`), on earlier integrating variables for multidimensional domains, on
`PDESystem` parameters, or be infinite (`-Inf`/`Inf` bounds are mapped to a finite
`σ` interval exactly as in NeuralPDE 6). Inside the integrand the integrating
variables are appended to the network input, so shifted arguments (`u(t - τ)`),
inner derivatives (`Differential(τ)(u(τ))`) and nested `Integral` terms work as well.

The quadrature rule is fixed-node only so the lowered term is a static array
expression (Reactant compatible); it defaults to `Integrals.GaussLegendre`, inherits
`quadrature_alg` from `QuadratureTraining`, and can be set explicitly with the
`integral_alg` keyword of `PhysicsInformedNN`.

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
NeuralPDE.nn_vcat
NeuralPDE.nn_veccat
NeuralPDE.quadrature
NeuralPDE.QuadratureIntegrand
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
