# `PhysicsInformedNN` Discretizer for PDESystems

Using the PINNs solver, we can solve general nonlinear PDEs:

![generalPDE](https://user-images.githubusercontent.com/12683885/86625781-5648c800-bfce-11ea-9d99-fbcb5c37fe0c.png)

with suitable boundary conditions:

![bcs](https://user-images.githubusercontent.com/12683885/86625874-8001ef00-bfce-11ea-9417-1a216c7d90aa.png)

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

## `symbolic_discretize` and the lower-level interface

```@docs
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.PhysicsInformedNN)
NeuralPDE.PINNRepresentation
NeuralPDE.PINNLossFunctions
```
