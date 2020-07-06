# Solving ODEs with Neural Networks

```julia
using Flux
using DifferentialEquations
using LinearAlgebra
using Plots
using Statistics

NNODE = Chain(x -> [x],
           Dense(1, 32, tanh),
           Dense(32, 1),
           first)
NNODE(1.0)
g(t) = t * NNODE(t) + 1f0

ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t + ϵ) - g(t)) / ϵ) - cos(2π * t)) for t in 0:1f-2:1f0)
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () # callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss())
    end
end
display(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)

t = 0:0.001:1.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π .* t) / 2π, label="True")
```

## Parameters

### chain

The `chain` parameter defines a chain of layers as the neural network to approximate the ODE.

### opt

The optimizer, which defaults to the [BFGS method](<https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>, to use for training the neural net.

### init_params

The initial condition for the ODE system, which defaults to DiffEqFlux initialization.

For ODEs, [see the DifferentialEquations.jl documentation](http://docs.juliadiffeq.org/dev/solvers/ode_solve#NeuralNetDiffEq.jl-1)
for the `nnode(chain,opt=ADAM(0.1))` algorithm, which takes in a Flux.jl chain
and optimizer to solve an ODE. This method is not particularly efficient, but
is parallel. It is based on the work of:

[Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.](https://arxiv.org/pdf/physics/9705023.pdf)
