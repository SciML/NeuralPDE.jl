# Solving Random Ordinary Differential Equations

In this tutorial we will solve a RODE with `NNRODE`.
Consider the equation:

```math
du = f(u,p,t,W)dt
```

where ``f(u,p,t,W)=2u\sin(W)`` and ``W(t)`` is a Noise process.

```julia
f = (u,p,t,W) ->   2u*sin(W)
tspan = (0.00f0, 1.00f0)
u0 = 1.0f0
dt = 1/20f0
```
We start off by defining the `NoiseProcess` ``W(t)``. In this case, we define a simple Gaussian Process. See [Noise Processes](https://diffeq.sciml.ai/stable/features/noise_process/#noise_process-1) for defining other types of processes.

```julia
W = WienerProcess(0.0,0.0,nothing)
```

Then, we need to define our model. In order to define a model, we can use `Flux.chain` or `DiffEqFlux.FastChain`.

```julia
chain = Flux.Chain(Dense(2,5,elu),Dense(5,1)) #Model using Flux, GalacticFlux
```

```julia
chain = FastChain(FastDense(2,50,tanh), FastDense(50,2)) #Model using DiffEqFlux
```
And let's define our optimizer function:
```julia
opt = ADAM(1e-3)
```

Now, let's pass all the parameters to the algorithm and then call the solver. If we already have some initial parameters, we can pass them into the `NNRODE` as well.

```julia
alg = NNRODE(chain , W , opt , init_params)
```
```julia
sol = solve(prob, NeuralPDE.NNRODE(chain,W,opt), dt=dt, verbose = true,
            abstol=1e-10, maxiters = 15000)
```
