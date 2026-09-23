module NeuralPDEOptimizationReactantExt

# Weakdep-triggered extensions cannot be ordered by parallel precompilation;
# keep this marker out of the precompile queue. See NeuralPDEBPINNExt.
__precompile__(false)

# Marker extension: it is loaded exactly when OptimizationReactant is loaded
# alongside NeuralPDE, which is what `NeuralPDE.default_adtype()` checks via
# `Base.get_extension` to decide between `AutoReactant()` and `AutoEnzyme()`.

end
