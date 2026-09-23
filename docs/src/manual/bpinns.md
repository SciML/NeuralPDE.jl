# Bayesian PINN Solvers

Using the Bayesian PINN solvers, we can solve ODEs and PDEs and also simultaneously
perform parameter estimation on them.

!!! note "Loading the Bayesian PINN extension"

    The Bayesian PINN solvers (`BNNODE`, `ahmc_bayesian_pinn_ode`,
    `BayesianPINN`, `ahmc_bayesian_pinn_pde`) live in a package extension. To use
    them you must load `AdvancedHMC`, `MCMCChains` and `LogDensityProblems`
    alongside `NeuralPDE`:

    ```julia
    using ModelingToolkit, NeuralPDE, SciMLBase, AdvancedHMC, MCMCChains, LogDensityProblems
    ```

    Without those packages loaded, calling `BNNODE(...)` /
    `ahmc_bayesian_pinn_ode(...)` / `ahmc_bayesian_pinn_pde(...)` will raise a
    `MethodError` because the extension methods are not in scope.

The ODE solver `BNNODE` uses [`ahmc_bayesian_pinn_ode`](@ref) at a lower level.
PDE systems use [`BayesianPINN`](@ref) with [`ahmc_bayesian_pinn_pde`](@ref): the
discretizer lowers the `PDESystem` through the same residual `System` as
[`PhysicsInformedNN`](@ref), and each residual cost becomes a Gaussian likelihood
over its collocation batch for AdvancedHMC sampling.

```@docs
NeuralPDE.BNNODE
NeuralPDE.BayesianPINN
NeuralPDE.ahmc_bayesian_pinn_ode
NeuralPDE.ahmc_bayesian_pinn_pde
NeuralPDE.BPINNstats
NeuralPDE.BPINNsolution
```
