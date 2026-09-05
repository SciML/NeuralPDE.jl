# Debugging PINN Solutions

This page collects developer-facing hooks for inspecting NeuralPDE's
`PhysicsInformedNN` discretization path.

## Helper API

```@docs
NeuralPDE.build_loss_function
NeuralPDE.generate_training_sets
NeuralPDE.get_argument
NeuralPDE.get_bounds
NeuralPDE.get_loss_function
NeuralPDE.get_numeric_integral
NeuralPDE.get_variables
NeuralPDE.vector_to_parameters
```

## Inspecting a Discretization

Use `symbolic_discretize` when you want to inspect the pieces generated from a
`PDESystem` without immediately constructing an `OptimizationProblem`.

```julia
using DomainSets, Lux, ModelingToolkit, NeuralPDE, SciMLBase
import DomainSets: Interval

@parameters x
@variables u(..)
Dx = Differential(x)

eqs = [Dx(u(x)) ~ 0.0]
bcs = [u(0.0) ~ 0.0]
domains = [x in Interval(0.0, 1.0)]

chain = Chain(Dense(1, 8, tanh), Dense(8, 1))
discretization = PhysicsInformedNN(chain, GridTraining(0.1))

@named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
pinnrep = symbolic_discretize(pde_system, discretization)
```

The returned `PINNRepresentation` stores:

- `pde_indvars` and `bc_indvars`, the coordinate layouts used for each residual.
- `symbolic_pde_loss_functions` and `symbolic_bc_loss_functions`, the generated
  data-free residual kernels.
- `loss_functions`, the strategy-wrapped scalar objectives consumed by
  `Optimization.jl`.

## Inspecting Residual Kernels

The generated residual kernels are callable as:

```julia
residual = pinnrep.loss_functions.datafree_pde_loss_functions[1]
coords = reshape(collect(0.1:0.1:0.9), 1, :)
values = residual(coords, pinnrep.flat_init_params)
```

For grid, stochastic, and quasi-random training, `coords` is a `(D, N)` matrix:
rows are independent variables and columns are collocation points. For boundary
conditions under quadrature training, the coordinate matrix may contain only the
free boundary variables; fixed boundary coordinates are reconstructed by the
symbolic parser.

## Symbolic Parser Pipeline

`build_loss_function(pinnrep, eq, bc_indvars)` lowers one ModelingToolkit
equation into a batched residual kernel:

1. Build the residual as `expand_derivatives(eq.lhs - eq.rhs)`.
2. Walk the Symbolics/SymbolicUtils term tree.
3. Replace dependent-variable calls with `phi_eval`.
4. Replace derivatives of dependent variables with `deriv_fd`.
5. Replace PDE parameters with runtime parameter access or default values.
6. Replace integral terms with numeric quadrature callbacks.
7. Generate Julia code with `Symbolics.build_function(..., cse = true)`.
8. Dot-vectorize scalar arithmetic while preserving matrix-valued runtime calls.
9. Compile the result with `RuntimeGeneratedFunctions`.

The resulting residual values remain differentiable with respect to neural
network parameters through the network evaluations used inside the finite
difference stencil.

## Fast Regression Checks

The most focused parser checks live in:

- `test/Forward/forward__ode.jl`
- `test/Forward/forward__integral.jl`
- `test/Forward/forward__derivatives.jl`

These tests exercise direct generated residual evaluation before the cost of full
training examples is introduced.
