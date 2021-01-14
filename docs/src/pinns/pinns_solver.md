# Physics-Informed Neural Networks

Using the PINNs solver, we can solve general nonlinear PDEs:

![generalPDE](https://user-images.githubusercontent.com/12683885/86625781-5648c800-bfce-11ea-9d99-fbcb5c37fe0c.png)

 with suitable boundary conditions:

 ![bcs](https://user-images.githubusercontent.com/12683885/86625874-8001ef00-bfce-11ea-9417-1a216c7d90aa.png)

where time t is a special component of x, and Ω contains the temporal domain.

PDEs are defined using the ModelingToolkit.jl `PDESystem`:

```julia
pde_system = PDESystem(eq,bcs,domains,param,var)
```

Here, `eq` is the equation, `bcs` represents the boundary conditions, `param` is
the parameter of the equation (like `[x,y]`), and `var` represents variables (like `[u]`).

The `PhysicsInformedNN` discretizer is defined as:

```julia
discretization = PhysicsInformedNN(chain,
                                   strategy;
                                   init_params = nothing,
                                   phi = nothing,
                                   derivative = nothing,
                                   )
```

Keyword arguments:

- `chain` is a Flux.jl chain, where the input of NN equals the number of dimensions and output equals the number of equations in the system,
- `strategy` determines which training strategy will be used,
- `init_params` is the initial parameter of the neural network. If nothing then automatically generated from the neural network,
- `phi` is a trial solution,
- `derivative` is a method that calculates the derivative.

The method `discretize` interprets from the ModelingToolkit PDE form to the PINNs Problem.

```julia
prob = discretize(pde_system, discretization)
```

which outputs an `OptimizationProblem` for [GalacticOptim.jl](https://galacticoptim.sciml.ai/dev/).
