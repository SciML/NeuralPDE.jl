# `BayesianPINN` Discretizer for PDESystems

Using the Bayesian PINNs solvers, we can solve general nonlinear PDEs,ODEs and Also simultaniously perform PDE,ODE parameter Estimation.

Note: The BPINN PDE solver also works for ODEs defined using ModelingToolkit, [ModelingToolkit.jl PDESystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/systems/PDESystem/). Despite this the ODE specific BPINN solver `BNNODE` [refer](https://docs.sciml.ai/NeuralPDE/dev/manual/ode/#NeuralPDE.BNNODE) exists and uses `NeuralPDE.advancedhmc_pinn_ode` at a lower level.

# `BayesianPINN` Discretizer for PDESystems and lower level Bayesian PINN Solver calls for PDEs and ODEs.

```@docs
NeuralPDE.BayesianPINN
NeuralPDE.advancedhmc_pinn_pde
NeuralPDE.advancedhmc_pinn_ode
```

## `symbolic_discretize` for `BayesianPINN` and lower level interface.

```@docs
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.AbstractPINN)
NeuralPDE.BPINNstats
NeuralPDE.BPINNsolution
```

