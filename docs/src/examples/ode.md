# Solving ODEs with Neural Networks

```julia
using Flux, Optim
using NeuralNetDiffEq
# Run a solve on scalars
linear = (u, p, t) -> cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
@time sol = solve(prob, NeuralNetDiffEq.NNODE(chain, opt), dt=1 / 20f0, verbose=true,
            abstol=1e-10, maxiters=200)
```

## Parameters

### prob

ODEProblem takes a function, initial condition, and time spans.

### algo

The algorithm parameter takes a NNODE object with the chain and the optimizer.

### chain

The `chain` parameter defines a chain of layers as the neural network to approximate the ODE.

### opt

The optimizer, which defaults to the [BFGS method](<https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>, to use for training the neural net.
`opt = Flux.ADAM(0.1, (0.9, 0.95))` defines an ADAM optimizer.

### init_params

The initial condition for the ODE system, which defaults to DiffEqFlux initialization.

## DifferentialEquations.jl

For ODEs, [see the DifferentialEquations.jl documentation](http://docs.juliadiffeq.org/dev/solvers/ode_solve#NeuralNetDiffEq.jl-1)
for the `nnode(chain,opt=ADAM(0.1))` algorithm, which takes in a Flux.jl chain
and optimizer to solve an ODE. This method is not particularly efficient, but
is parallel. It is based on the work of:

[Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.](https://arxiv.org/pdf/physics/9705023.pdf)
