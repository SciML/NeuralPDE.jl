# Developer Interfaces

This page documents extension points for packages that build on NeuralPDE. These
interfaces are versioned developer APIs, not the recommended application-level
entry points. Users should prefer the concrete discretizations, algorithms, and
training strategies in the manual.

## PDE Discretizations

```@docs
NeuralPDE.AbstractPINN
```

The concrete subtype is the dispatch extension point for
`SciMLBase.symbolic_discretize`. Its method should translate the symbolic
`PDESystem` into the representation consumed by its training workflow. The
abstract type and application-facing examples are documented on the
[PINN manual page](@ref).

## Training Strategies

```@docs
NeuralPDE.AbstractTrainingStrategy
```

A custom training strategy implements the generic `NeuralPDE.get_loss_function`
interface, which is documented on the [developer debugging page](@ref). It
returns a callable scalar objective. The interval form receives lower and upper
bounds as separate arguments. `NeuralPDE.generate_training_sets` and
`NeuralPDE.get_bounds` are optional extension points for strategies that construct
grid or bound-based data.

```julia
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
