# Solving ODEs with Neural Networks

The following is an example of solving a DifferentialEquations.jl
`ODEProblem` with a neural network using the physics-informed neural
networks approach specialized to 1-dimensional PDEs (ODEs).

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
