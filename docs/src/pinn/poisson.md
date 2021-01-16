# 2D Poisson Equation

In this example, we will solve a Poisson equation of 2 dimensions:

![poisson](https://user-images.githubusercontent.com/12683885/86838505-ee1ae480-c0a8-11ea-8d3c-7da53a9a7091.png)

with the boundary conditions:

![boundary](https://user-images.githubusercontent.com/12683885/86621678-437ec500-bfc7-11ea-8fe7-23a46a524cbe.png)

on the space domain:

![spaces](https://user-images.githubusercontent.com/12683885/86621460-e8e56900-bfc6-11ea-9b64-826ac84c36c9.png)

with grid discretization `dx = 0.1`. We will use physics-informed neural networks.

## Copy-Pastable Code

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
@parameters x y
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

# Boundary conditions
bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]

# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain,GridTraining(dx))

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)

#Optimizer
opt = Optim.BFGS()

#Callback function
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=1000)
phi = discretization.phi

xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
```

## Detailed Description

The ModelingToolkit PDE interface for this example looks like this:

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
@parameters x y
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

# Boundary conditions
bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]
```

Here, we define the neural network, where the input of NN equals the number of dimensions and output equals the number of equations in the system.

```julia
# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
```

Here, we build PhysicsInformedNN algorithm where `dx` is the step of discretization and `strategy` stores information for choosing a training strategy.
```julia
# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx))
```

As described in the API docs, we now need to define the `PDESystem` and create PINNs problem using the `discretize` method.

```julia
pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)
```

Here, we define the callback function and the optimizer. And now we can solve the PDE using PINNs
(with the number of epochs `maxiters=1000`).

```julia
#Optimizer
opt = Optim.BFGS()

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=1000)
phi = discretization.phi
```

We can plot the predicted solution of the PDE and compare it with the analytical solution in order to plot the relative error.

```julia
xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
```
![poissonplot](https://user-images.githubusercontent.com/12683885/90962648-2db35980-e4ba-11ea-8e58-f4f07c77bcb9.png)
