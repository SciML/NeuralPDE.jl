# [Adaptive Loss Functions](@id adaptive_loss)

The NeuralPDE `discretize` function allows for specifying an adaptive loss function strategy
which improves training performance by reweighting the equations as necessary to ensure
the boundary conditions are well-satisfied, even in ill-conditioned scenarios.

Some strategies reweight by gradient magnitudes (`GradientScaleAdaptiveLoss`) or via an
inner optimiser (`MiniMaxAdaptiveLoss`). Others are gradient-free and reweight purely based
on loss values (`SoftAdaptAdaptiveLoss`, `ReLoBRaLoAdaptiveLoss`), making them cheaper
to apply at each step.

The following are the options for the `adaptive_loss` keyword argument:

```@docs
NeuralPDE.AbstractAdaptiveLoss
NeuralPDE.NonAdaptiveLoss
NeuralPDE.GradientScaleAdaptiveLoss
NeuralPDE.MiniMaxAdaptiveLoss
NeuralPDE.SoftAdaptAdaptiveLoss
NeuralPDE.ReLoBRaLoAdaptiveLoss
```
