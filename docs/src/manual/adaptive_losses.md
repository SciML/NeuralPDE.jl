# [Adaptive Loss Functions](@id adaptive_loss)

The NeuralPDE `discretize` function allows for specifying adaptive loss function strategy
which improve training performance by reweighing the equations as necessary to ensure
the boundary conditions are well-satisfied, even in ill-conditioned scenarios. The following
are the options for the `adaptive_loss`:

```@docs
NeuralPDE.NonAdaptiveLoss
NeuralPDE.GradientScaleAdaptiveLoss
NeuralPDE.MiniMaxAdaptiveLoss
```
