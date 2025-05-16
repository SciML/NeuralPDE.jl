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
[ModelingToolkit.jl PDESystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/systems/PDESystem/).

## The `PhysicsInformedNN` Discretizer

```@docs
NeuralPDE.PhysicsInformedNN
NeuralPDE.Phi
SciMLBase.discretize(::PDESystem, ::NeuralPDE.PhysicsInformedNN)
```

## `symbolic_discretize` for `PhysicsInformedNN` and the lower-level interface

```@docs
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.AbstractPINN)
NeuralPDE.PINNRepresentation
NeuralPDE.PINNLossFunctions
```
