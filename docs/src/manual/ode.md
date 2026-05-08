# ODE-Specialized Physics-Informed Neural Network (PINN) Solver

```@docs
NNODE
```

# Bayesian inference with PINNs

!!! note "Loading the Bayesian PINN extension"

    `BNNODE` is provided by the `NeuralPDEBPINNExt` package extension. To use it,
    load `AdvancedHMC`, `MCMCChains` and `LogDensityProblems` alongside `NeuralPDE`:

    ```julia
    using NeuralPDE, AdvancedHMC, MCMCChains, LogDensityProblems
    ```

```@docs
BNNODE
```
