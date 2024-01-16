# `BayesianPINN` Discretizer for PDESystems

Using the Bayesian PINN solvers, we can solve general nonlinear PDEs, ODEs and also simultaneously perform parameter estimation on them.

Note: The BPINN PDE solver also works for ODEs defined using ModelingToolkit, [ModelingToolkit.jl PDESystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/systems/PDESystem/). Despite this, the ODE specific BPINN solver `BNNODE` [refer](https://docs.sciml.ai/NeuralPDE/dev/manual/ode/#NeuralPDE.BNNODE) exists and uses `NeuralPDE.ahmc_bayesian_pinn_ode` at a lower level.

# `BayesianPINN` Discretizer for PDESystems and lower level Bayesian PINN Solver calls for PDEs and ODEs.

```@docs
NeuralPDE.BayesianPINN
NeuralPDE.ahmc_bayesian_pinn_ode
NeuralPDE.ahmc_bayesian_pinn_pde
```

## `symbolic_discretize` for `BayesianPINN` and lower level interface.

```@docs
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.AbstractPINN)
NeuralPDE.BPINNstats
NeuralPDE.BPINNsolution
```

