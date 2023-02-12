# Solving ODEs with Physics-Informed Neural Networks (PINNs)

!!! note
    
    It is highly recommended you first read the [solving ordinary differential
    equations with DifferentialEquations.jl tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/)
    before reading this tutorial.

This tutorial is an introduction to using physics-informed neural networks (PINNs)
for solving ordinary differential equations (ODEs). In contrast to the later
parts of this documentation which use the symbolic interface, here we will focus on
the simplified `NNODE` which uses the `ODEProblem` specification for the ODE.
Mathematically, the `ODEProblem` defines a problem:

```math
u' = f(u,p,t)
```

for ``t \in (t_0,t_f)`` with an initial condition ``u(t_0) = u_0``. With physics-informed
neural networks, we choose a neural network architecture `NN` to represent the solution `u`
and seek parameters `p` such that `NN' = f(NN,p,t)` for all points in the domain.
When this is satisfied sufficiently closely, then `NN` is thus a solution to the differential
equation.

## Solving an ODE with NNODE

Let's solve the simple ODE:

```math
u' = \cos(2\pi t)
```

for ``t \in (0,1)`` and ``u_0 = 0`` with `NNODE`. First, we define the `ODEProblem` as we would
with any other DifferentialEquations.jl solver. This looks like:

```@example nnode1
using NeuralPDE, Flux, OptimizationOptimisers

linear(u, p, t) = cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)
```

Now, to define the `NNODE` solver, we must choose a neural network architecture. To do this, we
will use the [Flux.jl library](https://fluxml.ai/) to define a multilayer perceptron (MLP)
with one hidden layer of 5 nodes and a sigmoid activation function. This looks like:

```@example nnode1
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
```

Now we must choose an optimizer to define the `NNODE` solver. A common choice is `ADAM`, with
a tunable rate , which we will set to `0.1`. In general, this rate parameter should be
decreased if the solver's loss tends to be unsteady (sometimes rise “too much”), but should be
as large as possible for efficiency. Thus, the definition of the `NNODE` solver is as follows:

```@example nnode1
opt = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, opt)
```

Once these pieces are together, we call `solve` just like with any other `ODEProblem` solver.
Let's turn on `verbose` so we can see the loss over time during the training process:

```@example nnode1
sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true, abstol = 1.0f-6,
            maxiters = 200)
```

And that's it: the neural network solution was computed by training the neural network and
returned in the standard DifferentialEquations.jl `ODESolution` format. For more information
on handling the solution, consult
[the DifferentialEquations.jl solution handling section](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/).
