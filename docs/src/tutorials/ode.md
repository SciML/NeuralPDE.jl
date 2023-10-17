# Solving ODEs with Physics-Informed Neural Networks (PINNs)

!!! note
    
    It is highly recommended you first read the [solving ordinary differential
    equations with DifferentialEquations.jl tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/)
    before reading this tutorial.

This tutorial is an introduction to using physics-informed neural networks (PINNs) for solving ordinary differential equations (ODEs). In contrast to the later parts of this documentation which use the symbolic interface, here we will focus on the simplified [`NNODE`](@ref) which uses the `ODEProblem` specification for the ODE.

Mathematically, the `ODEProblem` defines a problem:

```math
u' = f(u,p,t)
```

for ``t \in (t_0,t_f)`` with an initial condition ``u(t_0) = u_0``. With physics-informed neural networks, we choose a neural network architecture `NN` to represent the solution `u` and seek parameters `p` such that `NN' = f(NN,p,t)` for all points in the domain. When this is satisfied sufficiently closely, then `NN` is thus a solution to the differential equation.

## Solving an ODE with NNODE

Let's solve a simple ODE:

```math
u' = \cos(2\pi t)
```

for ``t \in (0,1)`` and ``u_0 = 0`` with [`NNODE`](@ref). First, we define an `ODEProblem` as we would for defining an ODE using DifferentialEquations.jl interface. This looks like:

```@example nnode1
using NeuralPDE

linear(u, p, t) = cos(t * 2 * pi)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0, tspan)
```

Now, to define the [`NNODE`](@ref) solver, we must choose a neural network architecture. To do this, we will use the [Lux.jl](https://lux.csail.mit.edu/) to define a multilayer perceptron (MLP) with one hidden layer of 5 nodes and a sigmoid activation function. This looks like:

```@example nnode1
using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)
chain = Chain(Dense(1, 5, σ), Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> Lux.f64
```

Now we must choose an optimizer to define the [`NNODE`](@ref) solver. A common choice is `Adam`, with a tunable rate, which we will set to `0.1`. In general, this rate parameter should be decreased if the solver's loss tends to be unsteady (sometimes rise “too much”), but should be as large as possible for efficiency. We use `Adam` from [OptimizationOptimisers](https://docs.sciml.ai/Optimization/stable/optimization_packages/optimisers/). Thus, the definition of the [`NNODE`](@ref) solver is as follows:

```@example nnode1
using OptimizationOptimisers

opt = Adam(0.1)
alg = NNODE(chain, opt, init_params = ps)
```

Once these pieces are together, we call `solve` just like with any other `ODEProblem`. Let's turn on `verbose` so we can see the loss over time during the training process:

```@example nnode1
sol = solve(prob, alg, verbose = true, maxiters = 2000, saveat = 0.01)
```

Now lets compare the predictions from the learned network with the ground truth which we can obtain by numerically solving the ODE.

```@example nnode1
using OrdinaryDiffEq, Plots

ground_truth = solve(prob, Tsit5(), saveat = 0.01)

plot(ground_truth, label = "ground truth")
plot!(sol.t, sol.u, label = "pred")
```

And that's it: the neural network solution was computed by training the neural network and returned in the standard DifferentialEquations.jl `ODESolution` format. For more information on handling the solution, consult [here](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/).
