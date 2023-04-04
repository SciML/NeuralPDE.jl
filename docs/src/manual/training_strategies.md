# Training Strategies

Training strategies are the choices for how the points are sampled for the definition
of the physics-informed loss.

## Recommendations

`QuasiRandomTraining` with its default `LatinHyperCubeSample()` is a well-rounded training
strategy which can be used for most situations. It scales well for high dimensional
spaces and is GPU-compatible. `QuadratureTraining` can lead to faster or more robust convergence
with one of the H-Cubature or P-Cubature methods, but are not currently GPU compatible.
For very high dimensional cases, `QuadratureTraining` with an adaptive Monte Carlo quadrature
method, such as `CubaVegas`, can be beneficial for difficult or stiff problems.

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
