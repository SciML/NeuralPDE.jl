# [Training Strategies](@id training_strategies)

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

## Resampling and quasi-Newton optimizers

`StochasticTraining` and `QuasiRandomTraining` (with `resampling = true`) redraw their
collocation points when [`resample!`](@ref) is called from a `solve` callback. The points
are drawn from the discretization's own random number generator (the `rng` passed to
[`PhysicsInformedNN`](@ref), not the global `Random.default_rng()`), so a seeded
discretization resamples reproducibly; passing `rng = Xoshiro(seed)` to [`resample!`](@ref)
draws deterministically and independently of the discretization's stream.

Quasi-Newton optimizers (`BFGS`/`LBFGS`) build up an approximation of the curvature of
the loss (the Hessian or its inverse) across iterations. Resampling the collocation
points replaces the objective the curvature was estimated for, so the accumulated
history is stale and the next quasi-Newton step can be worse than a fresh first-order
step. Resampling therefore belongs in a first-order phase: run `Adam` (or another
first-order optimizer) with a resampling callback, then switch to `BFGS`/`LBFGS` on
fixed collocation points for the final refinement, as in the `train` helper of the test
suite.

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
