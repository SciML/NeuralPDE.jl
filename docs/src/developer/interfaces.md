# Developer Interfaces

This page documents extension points for packages that build on NeuralPDE. These
interfaces are versioned developer APIs, not the recommended application-level
entry points. Users should prefer the concrete discretizations, algorithms, and
training strategies in the manual.

## PDE Discretizations

`PhysicsInformedNN` is a `PDEBase.AbstractOptimizationSystemDiscretization`. The
lowering from a `PDESystem` to a `ModelingToolkit.System` is split into small
functions that can be extended:

```@docs
NeuralPDE.AbstractDerivativeLowering
NeuralPDE.lower
NeuralPDE.lower_derivative
NeuralPDE.AdditionalLoss
```

A new derivative backend implements `lower_derivative` for its own
`AbstractDerivativeLowering` subtype. The lowered expression must be a symbolic array
expression over the collocation matrix of the residual block (see
[`NeuralPDE.ResidualBlock`](@ref)) so that the representation stays independent of the
number of collocation points.

## Training Strategies

```@docs
NeuralPDE.AbstractTrainingStrategy
```

A training strategy used with `PhysicsInformedNN` implements the collocation
interface documented on the [training strategies page](@ref "Training Strategies"):
`NeuralPDE.collocation_count` and `NeuralPDE.sample_points`, plus optionally
`NeuralPDE.resamples` and `NeuralPDE.uses_quadrature_weights`.

```julia
struct FixedPointsTraining <: NeuralPDE.AbstractTrainingStrategy end

NeuralPDE.collocation_count(::FixedPointsTraining, kind, ivpos, bounds, pinned) =
    isempty(ivpos) ? 1 : 3
function NeuralPDE.sample_points(::FixedPointsTraining, block::NeuralPDE.ResidualBlock, rng)
    lb, ub = block.bounds
    return lb .+ (ub .- lb) .* [0.25 0.5 0.75], nothing
end
```

A strategy used with the ODE solvers (`NNODE`, `NNDAE`, `NNSDE`) implements the
generic `NeuralPDE.get_loss_function` interface instead, which returns a callable scalar
objective:

```julia
using Statistics: mean

struct MyTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int
end

function NeuralPDE.get_loss_function(
        init_params, residual, training_data, T, strategy::MyTraining; kwargs...
    )
    return θ -> mean(abs2, residual(training_data, θ))
end
```

## ODE Algorithms

```@docs
NeuralPDE.NeuralPDEAlgorithm
```

A concrete algorithm extends `SciMLBase.__solve` for an
`SciMLBase.AbstractODEProblem`, returns a callable SciMLBase solution, and
declares complex-number support with `SciMLBase.allowscomplex` when applicable.
