@verbosity_specifier NeuralPDEVerbosity begin
    toggles = (:training_progress,)
    presets = (
        None = (training_progress = Silent(),),
        Minimal = (training_progress = Silent(),),
        Standard = (training_progress = Silent(),),
        Detailed = (training_progress = InfoLevel(),),
        All = (training_progress = InfoLevel(),),
    )
    groups = ()
end

@doc """
    NeuralPDEVerbosity <: AbstractVerbositySpecifier

Verbosity specifier for the `verbose` keyword of `solve` with [`NNODE`](@ref),
[`NNDAE`](@ref), [`NNSDE`](@ref) and [`PINOODE`](@ref).

## Toggles

  - `training_progress`: report the loss on optimizer callbacks. An optimizer may call
    back more than once for the same iteration. This toggle is `Silent()` in the
    `None`, `Minimal` and `Standard` presets and `InfoLevel()` in `Detailed` and `All`.
    The numeric severity must fit in `Int32`.
    Passing `ErrorLevel()` for this toggle causes `solve` to throw an
    `ArgumentError` before training setup.

## Constructors

    NeuralPDEVerbosity(preset::Union{None, Minimal, Standard, Detailed, All})
    NeuralPDEVerbosity(; preset = nothing, training_progress = ...)

Both constructors accept the five SciMLLogging presets listed above. The keyword
constructor defaults to `Standard()`. The `training_progress` keyword overrides
the selected preset's level.

## Accepted `verbose` values

  - omitted or `false`: no training progress.
  - `true`: training progress at `InfoLevel()`.
  - A `NeuralPDEVerbosity` or one of `None()`, `Minimal()`, `Standard()`,
    `Detailed()` or `All()`. Other preset types raise an `ArgumentError`.
  - Any other `AbstractVerbositySpecifier`, such as the `DEVerbosity` that
    `DiffEqBase.solve` passes when `verbose` is omitted, configures other packages' messages
    and is treated as `NeuralPDEVerbosity()`.
""" NeuralPDEVerbosity

function neuralpde_verbosity(verbose::NeuralPDEVerbosity)
    verbose.training_progress == ErrorLevel() && throw(
        ArgumentError("ErrorLevel() throws on emission. Choose a non-throwing message level for training_progress.")
    )
    severity = verbosity_to_int(verbose.training_progress)
    typemin(Int32) <= severity <= typemax(Int32) ||
        throw(ArgumentError("training_progress log severity must fit in Int32."))
    return verbose
end

function neuralpde_verbosity(preset::Union{None, Minimal, Standard, Detailed, All})
    return NeuralPDEVerbosity(preset)
end

function neuralpde_verbosity(preset::AbstractVerbosityPreset)
    throw(
        ArgumentError(
            "Unsupported verbosity preset $(nameof(typeof(preset))). Use None(), Minimal(), Standard(), Detailed(), or All()."
        )
    )
end

function neuralpde_verbosity(verbose::Bool)
    return verbose ? NeuralPDEVerbosity(; training_progress = InfoLevel()) :
        NeuralPDEVerbosity(None())
end
neuralpde_verbosity(::AbstractVerbositySpecifier) = NeuralPDEVerbosity()
