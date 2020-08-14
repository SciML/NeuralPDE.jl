# Physics-Informed Neural Networks solver

## Example 1: Solving the 2-dimensional Poisson Equation

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
![poissonplot](https://user-images.githubusercontent.com/12683885/88482882-cbc00c80-cf6c-11ea-91bb-47a477f38af6.png)

## Example 2 : Solving the 2-dimensional Wave Equation with Neumann boundary condition

Let's solve this 2d wave equation:

![wave](https://user-images.githubusercontent.com/12683885/89735726-3c871e80-da6d-11ea-9c0b-21b5e09a97c3.png)

with grid discretization `dx = 0.1`.

Further, the solution of this equation with the given boundary conditions is presented.

```julia
@parameters x, t, θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t

#2D PDE
C=1
eq  = Dtt(u(x,t,θ)) ~ C^2*Dxx(u(x,t,θ))

# Initial and boundary conditions
bcs = [u(0,t,θ) ~ 0.,# for all t > 0
       u(1,t,θ) ~ 0.,# for all t > 0
       u(x,0,θ) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(x,0,θ)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
discretization = NeuralPDE.PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.02)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)
phi,res  = solve(prob,alg,verbose=true, maxiters=5000)
```

We can plot the predicted solution of the PDE and compare it with the analytical solution in order to plot the relative error.

```julia
xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_sol_func(x,t) =  sum([(8/(k^3*pi^3)) * sin(k*pi*x)*cos(C*k*pi*t) for k in 1:2:50000])

u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))

diff_u = abs.(u_predict .- u_real)
p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
p2 =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
```
![waveplot](https://user-images.githubusercontent.com/12683885/89735816-c7681900-da6d-11ea-847a-ae5b1cb52af9.png)

## Example 3 : Solving System of PDE and Matrix PDEs form

In this example we will solve the linear PDE system

![pdesystem](https://user-images.githubusercontent.com/12683885/89735129-fe87fb80-da68-11ea-936d-62311dde9bbc.png)

with grid discretization `dx = 0.1`.

```julia
@parameters x, y, θ
@variables u1(..), u2(..)
@derivatives Dx'~x
@derivatives Dy'~y

# System of pde
eqs = [Dx(u1(x,y,θ)) + 4*Dy(u2(x,y,θ)) ~ 0,
      Dx(u2(x,y,θ)) + 9*Dy(u1(x,y,θ)) ~ 0]

# Initial and boundary conditions
bcs = [[u1(x,0,θ) ~ 2x, u2(x,0,θ) ~ 3x]]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0), y ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.1
discretization = NeuralPDE.PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(2,8,Flux.σ),FastDense(8,2))

pde_system = PDESystem(eqs,bcs,domains,[x,y],[u1,u2])
prob = NeuralPDE.discretize(pde_system,discretization)
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)
phi,res = solve(prob,alg,verbose=true, maxiters=500)
```
### Matrix PDEs form
Also, in addition to systems, we can use the matrix form of PDEs:

```julia
@parameters x y θ
@variables u[1:2,1:2](..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# matrix PDE
eqs  = @. [(Dxx(u_(x,y,θ)) + Dyy(u_(x,y,θ))) for u_ in u] ~ -sin(pi*x)*sin(pi*y)*[0 1; 0 1]

# Initial and boundary conditions
bcs = [[u[1](x,0,θ) ~ x u[2](x,0,θ) ~ 2;
        u[3](x,0,θ) ~ 3 u[4](x,0,θ) ~ 4]]
```

## Example 4 : Solving ODE with 3rd order derivative

Let's consider ODE with 3rd order derivative:

![hdode](https://user-images.githubusercontent.com/12683885/89736407-dc46ab80-da71-11ea-9c6e-5964488642bd.png)

with grid discretization `dx = 0.1`.

```julia
@parameters x θ
@variables u(..)
@derivatives Dxxx'''~x
@derivatives Dx'~x

#ODE
eq = Dxxx(u(x,θ)) ~ cos(pi*x)

# Initial and boundary conditions
bcs = [u(0.,θ) ~ 0.0,
       u(1.,θ) ~ cos(pi),
       Dx(u(1.,θ)) ~ 1.0]


# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.1
discretization = NeuralPDE.PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.01)
chain = FastChain(FastDense(1,8,Flux.σ),FastDense(8,1))

pde_system = PDESystem(eq,bcs,domains,[x],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)
phi,res = solve(prob,alg,verbose=true, maxiters=1000)
```

We can plot the predicted solution of the PDE and its analytical solution.

```julia
analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)
xs = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

x_plot = collect(xs)
plot(x_plot ,u_real,label = "real" )
plot!(x_plot ,u_predict, label = "predict")
```
![hodeplot](https://user-images.githubusercontent.com/12683885/90276340-69bc3e00-de6c-11ea-89a7-7d291123a38b.png)
