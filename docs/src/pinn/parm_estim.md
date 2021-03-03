# Optimising Parameters of a Lorenz System
 Consider a Lorenz System ,
! [lorenzSystem](https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/lorenz_system.png) with Physics-Informed Neural Networks.
Now we would consider the case when we know σ = 10 and we want to optimise β  and ρ.
We start by defining the the problem,
```julia
@parameters t , β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ 10*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β*z(t)]


bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ IntervalDomain(0.0,1.0)]
dt = 0.1

```
And then the neural networks as,
```julia
input_ = length(domains)
n = 16
chain1 = FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
chain2 = FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
chain3 = FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
```
We will add an additional loss term based on the data that we have in order to optimise the parameters.
Here we simply calculate the solution of the lorenz system with [OrdinaryDiffEq.jl](https://diffeq.sciml.ai/v1.10/tutorials/ode_example.html#In-Place-Updates-1) and pick up some points randomly

```julia
function lorenz!(du,u,p,t)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,1.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob, Tsit5(), dt=0.1)
data = []
data
indx = rand(1:1:21 , 11)
for i in indx
    data = vcat(data , (sol.u[i] , sol.t[i]))
end

```
Then we define the additional loss function `additional_loss(phi, θ , p)`, the function has three arguments, `phi` the trial solution, `θ` the parameters of neural networks, and `p` optional parameters.

```julia
function additional_loss(phi, θ , p)
	l = length(θ) - 2
    _loss(u , t; t_ = cu[t]) = sum(abs2, phi[i](t_ , θ[(i*l - l + 1):(i*l)])[1] - (u[i])  for i in 1:1:3)
    return sum(abs2, _loss(u,t) for (u , t) in data)/length(data)
end
```

```julia
discretization = NeuralPDE.PhysicsInformedNN([chain1 , chain2, chain3],NeuralPDE.GridTraining(dt), param_estim=true, additional_loss=additional_loss)
pde_system = PDESystem(eqs,bcs,domains,[t],[x, y, z],[ρ, β], [1.0 ,1.0])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=4000)
p_ = res.minimizer[end-1:end]

```
