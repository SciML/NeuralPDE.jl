# `BayesianPINN` Discretizer for PDESystems

Using the Bayesian PINN solvers, we can solve general nonlinear PDEs, ODEs and also simultaneously perform parameter estimation on them.

!!! note "Loading the Bayesian PINN extension"

    The Bayesian PINN solvers (`BNNODE`, `ahmc_bayesian_pinn_ode`,
    `ahmc_bayesian_pinn_pde`) live in a package extension. To use them you must
    load `AdvancedHMC`, `MCMCChains` and `LogDensityProblems` alongside
    `NeuralPDE`:

    ```julia
    using NeuralPDE, AdvancedHMC, MCMCChains, LogDensityProblems
    ```

    Without those packages loaded, calling `BNNODE(...)` /
    `ahmc_bayesian_pinn_ode(...)` / `ahmc_bayesian_pinn_pde(...)` will raise a
    `MethodError` because the extension methods are not in scope.

Note: The BPINN PDE solver also works for ODEs defined using ModelingToolkit, [ModelingToolkit.jl PDESystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/API/PDESystem/). Despite this, the ODE specific BPINN solver `BNNODE` [refer](https://docs.sciml.ai/NeuralPDE/dev/manual/ode/#NeuralPDE.BNNODE) exists and uses `NeuralPDE.ahmc_bayesian_pinn_ode` at a lower level.

# `BayesianPINN` Discretizer for PDESystems and lower level Bayesian PINN Solver calls for PDEs and ODEs.

```@docs
NeuralPDE.BayesianPINN
NeuralPDE.ahmc_bayesian_pinn_ode
NeuralPDE.ahmc_bayesian_pinn_pde
```

## `symbolic_discretize` for `BayesianPINN` and lower level interface.

```@docs; canonical=false
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.AbstractPINN)
```

```@docs
NeuralPDE.BPINNstats
NeuralPDE.BPINNsolution
```
