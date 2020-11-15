# Physics-Informed Neural Networks

Using the PINNs solver, we can solve general nonlinear PDEs:

![generalPDE](https://user-images.githubusercontent.com/12683885/86625781-5648c800-bfce-11ea-9d99-fbcb5c37fe0c.png)

 with suitable boundary conditions:

 ![bcs](https://user-images.githubusercontent.com/12683885/86625874-8001ef00-bfce-11ea-9417-1a216c7d90aa.png)

where time t is a special component of x, and Ω contains the temporal domain.

We describe the PDE in the form of the ModelingToolKit interface. See an example of how this can be done above or take a look at the tests.

A General PDE Problem can be defined using a `PDESystem`:

```julia
pde_system = PDESystem(eq,bcs,domains,param,var)
```

Here, `eq` is the equation, `bcs` represents the boundary conditions,
`param` is the parameter of the equation (like `[x,y]`), and `var` represents variables (like `[u]`).

To solve this problem, use the `PhysicsInformedNN` algorithm.

```julia
discretization = PhysicsInformedNN(dx,
                                   chain,
                                   init_params = nothing;
                                   phi = nothing,
                                   autodiff=false,
                                   derivative = nothing,
                                   strategy = GridTraining())
```

Here,
- `dx` is a discretization of the grid
- `chain` is a Flux.jl chain, where the input of NN equals the number of dimensions and output equals the number of equations in the system
- `init_params` is the initial parameter of the neural network
- `phi` is a trial solution
- `autodiff` is a boolean variable that determines for the PDE operators whether to use automatic differentiation (not supported while) or numerical. The reverse mode of the loss function is always AD.
- `derivative` is a method that calculates the derivative
- `strategy` determines which training strategy will be used.

The method `discretize` interprets from the ModelingToolkit PDE form to the PINNs Problem.

```julia
prob = discretize(pde_system, discretization)
```

To run solve, we can use:
```julia
res = GalacticOptim.solve(prob, opt;  cb = cb, maxiters=maxiters)
```
Here,
`opt` is an optimizer, `cb` is a callback function, and `maxiters` is a number of iterations.


### Training strategy

List of training strategies that are available now:

 - `GridTraining()`: Initialize points on a lattice and never change them during
the training process.
 - `StochasticTraining()`: In each optimization iteration, we randomly select
the subset of points from a full training set.


### Low-level API

Besides the high-level API: `discretize(pde_system, discretization)`, we can also use the low-level API methods: `build_loss_function`, `get_loss_function` ,`generate_training_sets`,
`get_phi`, `get_derivative`.

See how this can be used in the docs examples or take a look at the tests.
