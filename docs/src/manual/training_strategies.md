# Training Strategies

Training strategies are the choices for how the points are sampled for the definition
of the physics-informed loss.

For `PhysicsInformedNN`, the collocation points of every equation and boundary condition
are parameters of the generated `OptimizationProblem`. The strategy decides how many
points each equation gets, how they are drawn and how the pointwise residuals are
reduced into a cost:

  - `GridTraining`: tensor-product grid, mean of the squared residuals. Interior points of
    the PDEs exclude the coordinate values pinned by the boundary conditions.
  - `StochasticTraining`: uniformly distributed random points, mean of the squared
    residuals. Call [`resample!`](@ref) from a `solve` callback to draw new points during
    training.
  - `QuasiRandomTraining`: quasi-Monte Carlo points, mean of the squared residuals;
    [`resample!`](@ref) redraws when `resampling = true`.
  - `QuadratureTraining`: tensor-product Gauss–Legendre nodes (`quadrature_alg =
    GaussLegendre(n = ...)`), quadrature of the squared residual normalized by the domain
    measure.

## Recommendations

`QuasiRandomTraining` with its default `LatinHyperCubeSample()` is a well-rounded training
strategy which can be used for most situations. It scales well for high dimensional
spaces and is GPU-compatible. `QuadratureTraining` can lead to faster or more robust
convergence on smooth low-dimensional problems.

`GridTraining` should only be used for testing purposes and should not be relied upon for real
training cases. `StochasticTraining` achieves a lower convergence rate in the quasi-Monte Carlo
methods and thus `QuasiRandomTraining` should be preferred in most cases. `WeightedIntervalTraining` can only be used with ODEs (`NNODE`).

## API

```@docs
GridTraining
StochasticTraining
QuasiRandomTraining
QuadratureTraining
WeightedIntervalTraining
```

## Extending the collocation interface

```@docs
NeuralPDE.collocation_count
NeuralPDE.sample_points
NeuralPDE.resamples
NeuralPDE.uses_quadrature_weights
```
