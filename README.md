# NeuralNetDiffEq

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/SciML/NeuralNetDiffEq.jl.svg?branch=master)](https://travis-ci.org/SciML/NeuralNetDiffEq.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v0eop301bx105av4?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/neuralnetdiffeq-jl)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/NeuralNetDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/NeuralNetDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/NeuralNetDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/NeuralNetDiffEq.jl?branch=master)

The repository is for the development of neural network solvers of differential equations such as physics-informed
neural networks (PINNs) and deep BSDE solvers. It utilizes techniques like deep neural networks and neural
stochastic differential equations to make it practical to solve high dimensional PDEs efficiently through the
likes of scientific machine learning (SciML).

# Examples

## DeepBSDE Solver

## Solving a 100 dimensional Hamilton-Jacobi-Bellman Equation

In this example we will solve a Hamilton-Jacobi-Bellman equation of 100 dimensions.
The Hamilton-Jacobi-Bellman equation is the solution to a stochastic optimal
control problem. Here, we choose to solve the classical Linear Quadratic Gaussian
(LQG) control problem of 100 dimensions, which is governed by the SDE
`dX_t = 2sqrt(λ)c_t dt + sqrt(2)dW_t` where `c_t` is a control process. The solution
to the optimal control is given by a PDE of the form:

![HJB](https://user-images.githubusercontent.com/1814174/63213366-b1817b80-c0d9-11e9-99b2-c8c08b86d2d5.PNG)

with terminating condition `g(X) = log(0.5f0 + 0.5f0*sum(X.^2))`. To solve it
using the `TerminalPDEProblem`, we write:

```julia
d = 100 # number of dimensions
X0 = fill(0.0f0,d) # initial value of stochastic control process
tspan = (0.0f0, 1.0f0)
λ = 1.0f0

g(X) = log(0.5f0 + 0.5f0*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = -λ*sum(σᵀ∇u.^2)
μ_f(X,p,t) = zero(X)  #Vector d x 1 λ
σ_f(X,p,t) = Diagonal(sqrt(2.0f0)*ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, X0, tspan)
```

As described in the API docs, we now need to define our `NNPDENS` algorithm
by giving it the Flux.jl chains we want it to use for the neural networks.
`u0` needs to be a `d` dimensional -> 1 dimensional chain, while `σᵀ∇u`
needs to be `d+1` dimensional to `d` dimensions. Thus we define the following:

```julia
hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.01)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)
```

And now we solve the PDE. Here we say we want to solve the underlying neural
SDE using the Euler-Maruyama SDE solver with our chosen `dt=0.2`, do at most
100 iterations of the optimizer, 100 SDE solves per loss evaluation (for averaging),
and stop if the loss ever goes below `1f-2`.

```julia
@time ans = solve(prob, pdealg, verbose=true, maxiters=100, trajectories=100,
                            alg=EM(), dt=0.2, pabstol = 1f-2)

```

### Solving the 100 dimensional Black-Scholes-Barenblatt Equation

In this example we will solve a Black-Scholes-Barenblatt equation of 100 dimensions.
The Black-Scholes-Barenblatt equation is a nonlinear extension to the Black-Scholes
equation which models uncertain volatility and interest rates derived from the
Black-Scholes equation. This model results in a nonlinear PDE whose dimension
is the number of assets in the portfolio.

To solve it using the `TerminalPDEProblem`, we write:

```julia
d = 100 # number of dimensions
X0 = repeat([1.0f0, 0.5f0], div(d,2)) # initial value of stochastic state
tspan = (0.0f0,1.0f0)
r = 0.05f0
sigma = 0.4f0
f(X,u,σᵀ∇u,p,t) = r * (u - sum(X.*σᵀ∇u))
g(X) = sum(X.^2)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(sigma*X) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, X0, tspan)
```

As described in the API docs, we now need to define our `NNPDENS` algorithm
by giving it the Flux.jl chains we want it to use for the neural networks.
`u0` needs to be a `d` dimensional -> 1 dimensional chain, while `σᵀ∇u`
needs to be `d+1` dimensional to `d` dimensions. Thus we define the following:

```julia
hls  = 10 + d #hide layer size
opt = Flux.ADAM(0.001)
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)
```

And now we solve the PDE. Here we say we want to solve the underlying neural
SDE using the Euler-Maruyama SDE solver with our chosen `dt=0.2`, do at most
150 iterations of the optimizer, 100 SDE solves per loss evaluation (for averaging),
and stop if the loss ever goes below `1f-6`.

```julia
ans = solve(prob, pdealg, verbose=true, maxiters=150, trajectories=100,
                            alg=EM(), dt=0.2, pabstol = 1f-6)
```


## PINNs solver

## Solving the 2-dimensional Poisson Equation

In this example we will solve a Poisson equation of 2 dimensions:

![poisson](https://user-images.githubusercontent.com/12683885/86838505-ee1ae480-c0a8-11ea-8d3c-7da53a9a7091.png)

with boundary conditions:

![boundary](https://user-images.githubusercontent.com/12683885/86621678-437ec500-bfc7-11ea-8fe7-23a46a524cbe.png)

on the space domain:

![spaces](https://user-images.githubusercontent.com/12683885/86621460-e8e56900-bfc6-11ea-9b64-826ac84c36c9.png)

with grid discretization `dx = 0.1`.

The ModelingToolkit PDE interface for this example looks like this:

```julia
@parameters x y θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq  = Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

# Boundary conditions
bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
discretization = PhysicsInformedNN(dx)
```

Here, we define the neural network and optimizer, where the input of NN equals the number of dimensions and output equals the number of equations in the system.

```julia
# Neural network and optimizer
opt = Flux.ADAM(0.02)
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
```

As described in the API docs, we now need to define `NNDE` algorithm by giving it the Flux.jl chains we want it to use for the neural networks. Also, we need to define `PDESystem` and then pass it to the method `discretize`.

```julia
pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)
alg = NNDE(chain,opt,autodiff=false)
```

And now we can solve the PDE using PINNs. At do a number of epochs `maxiters=5000`.

```julia
phi,res  = solve(prob,alg,verbose=true, maxiters=5000)
```

# API Documentation

### Solving High Dimensional PDEs with Neural Networks

To solve high dimensional PDEs, first one should describe the PDE in terms of
the `TerminalPDEProblem` with constructor:

```julia
TerminalPDEProblem(g,f,μ_f,σ_f,X0,tspan,p=nothing)
```

which describes the semilinear parabolic PDE of the form:

![](https://user-images.githubusercontent.com/1814174/63212617-48980480-c0d5-11e9-9fec-0776117464c7.PNG)

with terminating condition `u(tspan[2],x) = g(x)`. These methods solve the PDE in
reverse, satisfying the terminal equation and giving a point estimate at
`u(tspan[1],X0)`. The dimensionality of the PDE is determined by the choice
of `X0`, which is the initial stochastic state.

To solve this PDE problem, there exists two algorithms:

- `NNPDENS(u0,σᵀ∇u;opt=Flux.ADAM(0.1))`: Uses a neural stochastic differential
  equation which is then solved by the methods available in DifferentialEquations.jl
  The `alg` keyword is required for specifying the SDE solver algorithm that
  will be used on the internal SDE. All of the other keyword arguments are passed
  to the SDE solver.
- `NNPDEHan(u0,σᵀ∇u;opt=Flux.ADAM(0.1))`: Uses the stochastic RNN algorithm
  [from Han](https://www.pnas.org/content/115/34/8505). Only applicable when
  `μ_f` and `σ_f` result in a non-stiff SDE where low order non-adaptive time
  stepping is applicable.

Here, `u0` is a Flux.jl chain with `d` dimensional input and 1 dimensional output.
For `NNPDEHan`, `σᵀ∇u` is an array of `M` chains with `d` dimensional input and
`d` dimensional output, where `M` is the total number of timesteps. For `NNPDENS`
it is a `d+1` dimensional input (where the final value is time) and `d` dimensional
output. `opt` is a Flux.jl optimizer.

Each of these methods has a special keyword argument `pabstol` which specifies
an absolute tolerance on the PDE's solution, and will exit early if the loss
reaches this value. Its defualt value is `1f-6`.

### Solving ODEs with Neural Networks

For ODEs, [see the DifferentialEquations.jl documentation](http://docs.juliadiffeq.org/dev/solvers/ode_solve#NeuralNetDiffEq.jl-1)
for the `nnode(chain,opt=ADAM(0.1))` algorithm, which takes in a Flux.jl chain
and optimizer to solve an ODE. This method is not particularly efficient, but
is parallel. It is based on the work of:

[Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.](https://arxiv.org/pdf/physics/9705023.pdf)

### Solving Kolmogorov Equations with Neural Networks

A Kolmogorov PDE is of the form :

![](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/KolmogorovPDE.png)

Considering S be a solution process to the SDE:

![](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/StochasticP.png)

then the solution to the Kolmogorov PDE is given as:

![](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/Solution.png)

A Kolmogorov PDE Problem can be defined using a `SDEProblem`:

```julia
SDEProblem(μ,σ,u0,tspan,xspan,d)
```

Here `u0` is the initial distribution of x. Here we define `u(0,x)` as the probability density function of `u0`.`μ` and `σ` are obtained from the SDE for the stochastic process above. `d` represents the dimenstions of `x`.
`u0` can be defined using `Distributions.jl`.

Another was of defining a KolmogorovPDE is using the `KolmogorovPDEProblem`.

```julia
KolmogorovPDEProblem(μ,σ,phi,tspan,xspan,d)
```

Here `phi` is the initial condition on u(t,x) when t = 0. `μ` and `σ` are obtained from the SDE for the stochastic process above. `d` represents the dimenstions of `x`.

To solve this problem use,

- `NNKolmogorov(chain, opt , sdealg)`: Uses a neural network to realise a regression function which is the solution for the linear Kolmogorov Equation.

Here, `chain` is a Flux.jl chain with `d` dimensional input and 1 dimensional output.`opt` is a Flux.jl optimizer. And `sdealg` is a high-order algorithm to calculate the solution for the SDE, which is used to define the learning data for the problem. Its default value is the classic Euler-Maruyama algorithm.

## Solving PDEs using PINNs solver
Using PINNs solver, we might solve general nonlinear PDEs:

![generalPDE](https://user-images.githubusercontent.com/12683885/86625781-5648c800-bfce-11ea-9d99-fbcb5c37fe0c.png)

 with suitable boundary conditions:

 ![bcs](https://user-images.githubusercontent.com/12683885/86625874-8001ef00-bfce-11ea-9417-1a216c7d90aa.png)

Where time t is as a special component of x, and Ω contains the temporal domain.

We describe the PDE in the form of the ModelingToolKit interface. See an example of how this can be done above or take a look at the tests.

General PDE Problem can be defined using a `PDESystem`:

```julia
PDESystem(eq,bcs,domains,param,var)
```

Here, `eq` is equation, `bcs` is boundary conditions, `param` is parameter of eqution (like `[x,y]`) and var is varibles (like `[u]`).

The method `discretize` do interpret from ModelingToolkit PDE form to the PINNs Problem.

```julia
discretize(pde_system, discretization)
```

To solve this problem use `NNDE` algorithm.

```julia
NNDE(chain,opt, autodiff=false)
```

Here, `chain` is a Flux.jl chain with d dimensional input and 1 dimensional output. `opt` is a Flux.jl optimizer. And `autodiff` is a boolean variable that determines whether to use automatic differentiation(not supported while) or numerical.

## Related Packages

- [ReservoirComputing.jl](https://github.com/MartinuzziFrancesco/ReservoirComputing.jl) has an implementation of the [Echo State Network method](https://arxiv.org/pdf/1710.07313.pdf) for learning the attractor properties of a chaotic system.
