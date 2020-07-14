# Solving Random Ordinary Differential Equations
In this tutorial we will solve a RODE with `NNRODE` .
Consider the equation

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
We start off by defining the `NoiseProcess` ``W(t)`` . In this case we define a simple Gaussian Process. See [Noise Processes](https://docs.sciml.ai/stable/features/noise_process/#noise_process-1) for defining other types of process.

```julia
W = WienerProcess(0.0,0.0,nothing)
```
Then we need to define our model, in order to define a model we can use `Flux.chain` or `DiffEqFlux.FastChain`

```julia
chain = Flux.Chain(Dense(2,5,elu),Dense(5,1)) #Model using Flux
```

```julia
chain = FastChain(FastDense(2,50,tanh), FastDense(50,2)) #Model using DiffEqFlux
```
And lets define our optimiser function :
```julia
opt = ADAM(1e-3)
```

Now lets pass all parameters to the algorithm and then call the solver. If we already have some initial parameters we can pass them into the `NNRODE` as well.

```julia
alg = NNRODE(chain , W , opt , init_params)
```
```julia
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain,W,opt), dt=dt, verbose = true,
            abstol=1e-10, maxiters = 15000)
```
Now in-order to get the`W` from solution we can simply use a `NoiseWrapper` See this to know more about [Noise Wrapper](https://docs.sciml.ai/release-4.6/features/noise_process.html#Adaptive-NoiseWrapper-Example-1).
```julia
W2 = NoiseWrapper(sol.W)
```
