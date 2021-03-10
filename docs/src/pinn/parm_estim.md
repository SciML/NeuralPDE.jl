# Optimising Parameters of a Lorenz System
 Consider a Lorenz System ,

![lorenzSystem](https://user-images.githubusercontent.com/43771652/110070232-8172f980-7d9f-11eb-9d18-f1cf7e89c857.png)

with Physics-Informed Neural Networks.
Now we would consider the case where we want to optimise the parameters σ, β  and ρ.
We start by defining the the problem,

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, OrdinaryDiffEq
@parameters t ,σ_ ,β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β*z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ IntervalDomain(0.0,1.0)]
dt = 0.05
```
And the neural networks as,
```julia
input_ = length(domains)
n = 16
chain1 = FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
chain2 = FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
chain3 = FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
```
We will add an additional loss term based on the data that we have in order to optimise the parameters.

Here we simply calculate the solution of the lorenz system with [OrdinaryDiffEq.jl](https://diffeq.sciml.ai/v1.10/tutorials/ode_example.html#In-Place-Updates-1) based on the adaptivity of the ODE solver. This is used to introduce non-uniformity to the time series.

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
indx = rand(1:1:21 , 12)
for i in indx
    data = vcat(data , (sol.u[i] , sol.t[i]))
end
```
Then we define the additional loss funciton `additional_loss(phi, θ , p)`, the function has three arguments, `phi` the trial solution, `θ` the parameters of neural networks, and the hyperparameters `p` .

```julia
function additional_loss(phi, θ , p)
    l = Int(length(θ)/3)
    _loss(u , t) = sum(abs2, phi[i](t , θ[(i*l - l + 1):(i*l)])[1] - (u[i])  for i in 1:1:3)
    global data
    return sum(abs2, _loss(u,t) for (u , t) in data)/length(data)
end
```
Then finally defining and optimising using the `PhysicsInformedNN` interface.
```julia
discretization = NeuralPDE.PhysicsInformedNN([chain1 , chain2, chain3],NeuralPDE.GridTraining(dt), param_estim=true, additional_loss=additional_loss)
pde_system = PDESystem(eqs,bcs,domains,[t],[x, y, z],[σ_, ρ, β], [1.0, 1.0 ,1.0])
prob = NeuralPDE.discretize(pde_system,discretization)
cb = function (p,l)
    println("Current loss is: $l")
    return false
end
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=5000)
p_ = res.minimizer[end-2:end] # p_ = [9.93, 28.002, 2.667]
```
And then finally some analyisis by plotting.
```julia
initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
u_predict  = [[discretization.phi[i]([t],minimizers[i])[1] for t in sol.t] for i in 1:3]
plot(sol)
plot!(sol.t, u_predict, label = ["x(t)" "y(t)" "z(t)"])
```

![Plot_Lorenz](https://user-images.githubusercontent.com/43771652/110070388-e75f8100-7d9f-11eb-90ed-a165993e901e.png)
