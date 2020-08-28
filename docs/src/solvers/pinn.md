# Physics-Informed Neural Networks

Using PINNs solver, we might solve general nonlinear PDEs:

![generalPDE](https://user-images.githubusercontent.com/12683885/86625781-5648c800-bfce-11ea-9d99-fbcb5c37fe0c.png)

 with suitable boundary conditions:

 ![bcs](https://user-images.githubusercontent.com/12683885/86625874-8001ef00-bfce-11ea-9417-1a216c7d90aa.png)

Where time t is as a special component of x, and Ω contains the temporal domain.

We describe the PDE in the form of the ModelingToolKit interface. See an example of how this can be done above or take a look at the tests.

General PDE Problem can be defined using a `PDESystem`:

```julia
pde_system = PDESystem(eq,bcs,domains,param,var)
```

Here, `eq` is equation, `bcs` is boundary conditions, `param` is parameter of equation (like `[x,y]`) and `var` is varibles (like `[u]`).

To solve this problem use `PhysicsInformedNN` algorithm.

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
`dx` is a discretization of the grid,
`chain` is a Flux.jl chain, where the input of NN equals the number of dimensions and output equals the number of equations in the system.
`init_params` is the initial parameter of the neural network,
`phi` is a trial solution,
`autodiff` is a boolean variable that determines whether to use automatic, differentiation(not supported while) or numerical,
`derivative` is a method that calculates derivative,
`strategy` determines which training strategy will be used.

The method `discretize` do interpret from ModelingToolkit PDE form to the PINNs Problem.

```julia
prob = discretize(pde_system, discretization)
```

To run solve we can use:
```julia
res = GalacticOptim.solve(prob, opt;  cb = cb, maxiters=maxiters)
```
Here,
`opt` is an optimizer, `cb` is a callback function and `maxiters` is a number of iteration.


### Training strategy

List of training strategies that are available now:

 - `GridTraining()` : Initialize points on a lattice and never change them during
the training process.
 - `StochasticTraining()` :  In each optimization iteration, we select randomly
the subset of points from a full training set.


### Low-level API

Besides the high level of API: `discretize(pde_system, discretization)`, we can also use low-level API methods:  `build_loss_function`, `get_loss_function` ,`generate_training_sets`,`get_phi`, `get_derivative`.

See how this can be used in docs examples or take a look at the tests.

### GPUs

Actually, it is just enough add `|>gpu` after `chain` and all will works.
