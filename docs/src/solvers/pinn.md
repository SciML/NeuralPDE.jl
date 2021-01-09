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
                                   initθ = nothing,
                                   phi = nothing,
                                   derivative = nothing,
                                   )
```

Keyword arguments:

- `chain` is a Flux.jl chain, where the input of NN equals the number of dimensions and output equals the number of equations in the system,
- `strategy` determines which training strategy will be used,
- `initθ` is the initial parameter of the neural network. If nothing then automatically generated from the neural network,
- `phi` is a trial solution,
- `derivative` is a method that calculates the derivative.

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
 - `QuasiRandomTraining(points;sampling_alg = UniformSample(),minibatch=500)`: The training set is generated on quasi-random low discrepency sequences.`minibatch` is the number of subsets, where `points` is the number of quasi-random points in minibatch.
  The number of the total points is `length(lb) * points * minibatch`,
  where `lb` is the lower bound and `length(lb)` is the dimensionality.
  `sampling_alg` is the quasi-Monte Carlo sampling algorithm.
  On each iteration of training, it is randomly selected one of the minibatch.
  See the [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) for the full set of quasi-random sampling algorithms which are available.
- `QuadratureTraining(;quadrature_alg=HCubatureJL(),reltol= 1e-6,abstol= 1e-3,maxiters=1e3,batch=0)`: The loss is computed as an approximation of the integral of the PDE loss
  at each iteration using [adaptive quadrature methods](https://en.wikipedia.org/wiki/Adaptive_quadrature)
  via the differentiable [Quadrature.jl](https://github.com/SciML/Quadrature.jl).
  - `quadrature_alg` is quadrature algorithm,
  - `reltol`: relative tolerance,
  - `abstol`: absolute tolerance,
  - `maxiters`: the maximum number of iterations in quadrature algorithm,
  - `batch`: the preferred number of points to batch. If `batch` = 0, the number of points in the batch is determined adaptively by the algorithm.

  See the [Quadrature.jl](https://github.com/SciML/Quadrature.jl) documentation for the choices of quadrature methods.

### Low-level API

These additional methods exist to help with introspection:

- `symbolic_discretize(pde_system,discretization)`: This method is the same as `discretize` but instead
  returns the unevaluated Julia function to allow the user to see the generated training code.

See how this can be used in the docs examples or take a look at the tests.
