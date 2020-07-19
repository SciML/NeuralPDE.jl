# ODE-Specialized Physics Informed Neural Solver

The ODE-specialized physics-informed neural network (PINN) solver is a
method for the [DifferentialEquations.jl common interface](https://diffeq.sciml.ai/latest/)
of `ODEProblem` which generates the solution via a neural network.
Thus the standard [ODEProblem](https://diffeq.sciml.ai/latest/types/ode_types/)
is used, but a new algorithm, `NNODE` is utilized to solve the problem.

The algorithm type is:

```julia
nnode(chain,opt)
```

where `chain` is a DiffEqFlux `sciml_train` compatible Chain or FastChain
representing a neural network, and `opt` is an optimization method
for `sciml_train`. For more details, see [the DiffEqFlux documentation
on `sciml_train`](https://diffeqflux.sciml.ai/dev/).

[Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.](https://arxiv.org/pdf/physics/9705023.pdf)
