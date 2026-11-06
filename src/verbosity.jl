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

  - `training_progress`: the loss reported once per optimizer iteration. `Silent()` in the
    `None`, `Minimal` and `Standard` presets, `InfoLevel()` in `Detailed` and `All`.

## Constructors

    NeuralPDEVerbosity(preset::AbstractVerbosityPreset)
    NeuralPDEVerbosity(; preset = nothing, kwargs...)

Toggles are set by keyword, e.g. `NeuralPDEVerbosity(; training_progress = InfoLevel())`.
`NeuralPDEVerbosity()` uses the `Standard` preset.

## Accepted `verbose` values

  - omitted or `false`: no training progress.
  - `true`: training progress at `InfoLevel()`.
  - A `NeuralPDEVerbosity` or an `AbstractVerbosityPreset` such as `Detailed()`.
  - Any other `AbstractVerbositySpecifier`, such as the `DEVerbosity` that
    `DiffEqBase.solve` passes when `verbose` is omitted, configures other packages' messages
    and is treated as `NeuralPDEVerbosity()`.
""" NeuralPDEVerbosity

neuralpde_verbosity(verbose::NeuralPDEVerbosity) = verbose
neuralpde_verbosity(verbose::AbstractVerbosityPreset) = NeuralPDEVerbosity(verbose)
function neuralpde_verbosity(verbose::Bool)
    return verbose ? NeuralPDEVerbosity(; training_progress = InfoLevel()) :
        NeuralPDEVerbosity(None())
end
neuralpde_verbosity(::AbstractVerbositySpecifier) = NeuralPDEVerbosity()
