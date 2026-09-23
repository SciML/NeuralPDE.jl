# Adaptive loss weights

`discretize` can scalarize the costs of any multi-cost `System` with symbolic parameters.
The adaptive callbacks below update those parameters during `solve`, so the feature also
applies to optimization systems that are not PINNs.

## Build a weighted problem

Declare one symbolic parameter per cost and pass them to `discretize` as the `weights`
keyword. Supply an initial value for each weight through `p`:

```@example adaptive_loss
using DomainSets, Lux, NeuralPDE, OptimizationOptimisers
using ModelingToolkit: @parameters, @variables, Differential
import DomainSets: Interval

@parameters x w[1:3]
@variables u(..)
Dxx = Differential(x)^2
@named pdesys = PDESystem(
    [Dxx(u(x)) ~ -sinpi(x)], [u(0) ~ 0.0, u(1) ~ 0.0],
    [x ∈ Interval(0.0, 1.0)], [x], [u(x)], collect(w)
)
disc = PhysicsInformedNN(
    Chain(Dense(1, 16, tanh), Dense(16, 1)), GridTraining(0.1)
)
weight_symbols = collect(w)
prob = discretize(pdesys, disc; weights = weight_symbols,
    p = collect(weight_symbols .=> ones(3)))
```

Pass the problem and the same weight symbols to a callback constructor. The callback updates
only cost parameters; the optimizer continues to update the neural-network unknowns.

```@example adaptive_loss
callback = SoftAdaptAdaptiveLoss(prob, weight_symbols; every = 25, α = 0.1)
sol = solve(prob, Adam(0.01); callback, maxiters = 1)
```

## Available rules

`GradientScaleAdaptiveLoss` balances the maximum gradient magnitudes, `MiniMaxAdaptiveLoss`
ascends the weights with an Optimisers rule, `SoftAdaptAdaptiveLoss` responds to relative
loss changes, and `ReLoBRaLoAdaptiveLoss` combines random lookbacks with a moving average.
See each callback's docstring for its parameters and cited paper. Each accepts an
`OptimizationProblem` and the symbolic weight vector used to build it.
