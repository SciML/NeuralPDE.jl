# Solving the 2-dimensional Poisson Equation via Physics-Informed Neural Networks

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
bcs = [u(0,y,θ) ~ 0.f0, u(1,y,θ) ~ -sin(pi*1)*sin(pi*y),
       u(x,0,θ) ~ 0.f0, u(x,1,θ) ~ -sin(pi*x)*sin(pi*1)]
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
We can plot the predicted solution of the PDE and compare it with the analytical solution in order to plot the relative error.
```julia
xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

using Plots
p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
```
![image](https://user-images.githubusercontent.com/12683885/88294196-a29d4300-cd04-11ea-8e72-18867662583d.png)
