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
                                   nothing; #init_params
                                   phi = nothing,
                                   derivative = nothing,
                                   strategy = GridTraining())
```

Keyword arguments:

- `chain` is a Flux.jl chain, where the input of NN equals the number of dimensions and output equals the number of equations in the system
- `init_params` is the initial parameter of the neural network. If nothing then automatically generated from the neural network.
- `phi` is a trial solution
- `derivative` is a method that calculates the derivative
- `strategy` determines which training strategy will be used.

The method `discretize` interprets from the ModelingToolkit PDE form to the PINNs Problem.

```julia
prob = discretize(pde_system, discretization)
```

which outputs an `OptimizationProblem` for [GalacticOptim.jl](https://galacticoptim.sciml.ai/dev/).

### Training strategy

List of training strategies that are available now:

 - `GridTraining(dx)`: Initialize points on a lattice uniformly spaced via `dx`. If
   `dx` is a scalar, then `dx` corresponds to the spacing in each direction. If `dx`
   is a vector, then it should be sized to match the number of dimensions and corresponds
   to the spacing per direction.
 - `StochasticTraining(points)`: `points` number of sochastically sampled points from the domain. 
   In each optimization iteration, we randomly select a new subset of points from a full training set.
- `QuasiRandomTraining(sampling_alg)`: The training set is generated on quasi-random low discrepency
  sequences. See the [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) for the full
  set of quasi-random sampling algorithms which are available.
- `QuadratureTraining(quadrature_alg)`: The loss is computed as an approximation of the integral of the PDE loss 
  at each iteration using [adaptive quadrature methods](https://en.wikipedia.org/wiki/Adaptive_quadrature)
  via the differentiable [Quadrature.jl](https://github.com/SciML/Quadrature.jl). See the Quadrature.jl
  documentation for the choices of quadrature methods.

### Low-level API

These additional methods exist to help with introspection:

- `symbolic_discretize(pde_system,discretization)`: This method is the same as `discretize` but instead
  returns the unevaluated Julia function to allow the user to see the generated training code.

See how this can be used in the docs examples or take a look at the tests.
