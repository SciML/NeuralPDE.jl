# Physics-Informed Neural Networks solver

## Example 1: Solving the 2-dimensional Poisson Equation

In this example, we will solve a Poisson equation of 2 dimensions:

![poisson](https://user-images.githubusercontent.com/12683885/86838505-ee1ae480-c0a8-11ea-8d3c-7da53a9a7091.png)

with the boundary conditions:

![boundary](https://user-images.githubusercontent.com/12683885/86621678-437ec500-bfc7-11ea-8fe7-23a46a524cbe.png)

on the space domain:

![spaces](https://user-images.githubusercontent.com/12683885/86621460-e8e56900-bfc6-11ea-9b64-826ac84c36c9.png)

with grid discretization `dx = 0.1`.

The ModelingToolkit PDE interface for this example looks like this:

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
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
discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy = GridTraining())
```

As described in the API docs, we now need to define the `PDESystem` and create PINNs problem using the `discretize` method.

```julia
pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)
```

Here, we define the callback function and the optimizer. And now we can solve the PDE using PINNs
(with the number of epochs `maxiters=1000`).

```julia
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, BFGS(), progress = false; cb = cb, maxiters=1000)
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

## Example 2 : Solving the 2-dimensional Wave Equation with Neumann boundary condition

Let's solve this 2-dimensional wave equation:

![wave](https://user-images.githubusercontent.com/12683885/91465006-ecde8a80-e895-11ea-935e-2c1d60e3d1f2.png)


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

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy= GridTraining())

pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = discretize(pde_system,discretization)

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=1200)
phi = discretization.phi
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
![waveplot](https://user-images.githubusercontent.com/12683885/90979146-c9e16d00-e55b-11ea-93d7-fefa696008fd.png)



## Example 3 : Solving the 3-dimensional PDE

the 3-dimensional PDE:
![3dpde](https://user-images.githubusercontent.com/12683885/90976452-d2c74400-e545-11ea-8361-288603d9ddbc.png)

with the initial and boundary conditions:

![boundary](https://user-images.githubusercontent.com/12683885/91332936-8c881400-e7d5-11ea-991a-39c9d61d4f24.png)

on the space and time domain:

![space](https://user-images.githubusercontent.com/12683885/90976622-3605a600-e547-11ea-837e-92330769f5ee.png)

with grid discretization `dx = 0.25`, `dy = 0.25`, `dt = 0.25`.

```julia
# 3D PDE
@parameters x y t θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dt'~t

# 3D PDE
eq  = Dt(u(x,y,t,θ)) ~ Dxx(u(x,y,t,θ)) + Dyy(u(x,y,t,θ))
# Initial and boundary conditions
bcs = [u(x,y,0,θ) ~ exp(x+y)*cos(x+y) ,
       u(0,y,t,θ) ~ exp(y)*cos(y+4t)
       u(2,y,t,θ) ~ exp(2+y)*cos(2+y+4t) ,
       u(x,0,t,θ) ~ exp(x)*cos(x+4t),
       u(x,2,t,θ) ~ exp(x+2)*cos(x+2+4t)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]

# Discretization
dx = 0.25; dy= 0.25; dt = 0.25
# Neural network
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = NeuralPDE.PhysicsInformedNN([dx,dy,dt],
                                             chain,
                                             strategy = NeuralPDE.StochasticTraining(include_frac=0.9))
pde_system = PDESystem(eq,bcs,domains,[x,y,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1), progress = false; cb = cb, maxiters=3000)
phi = discretization.phi
```

## Example 4 : Solving a PDE System

In this example, we will solve the PDE system:

![pdesystem](https://user-images.githubusercontent.com/12683885/90978370-22157080-e556-11ea-92b3-d65cb9aa3115.png)

with the initial conditions:

![Initial](https://user-images.githubusercontent.com/12683885/90978670-322e4f80-e558-11ea-8157-a0b6ec84e121.png)

and the boundary conditions:

![boundary](https://user-images.githubusercontent.com/12683885/90978689-4c682d80-e558-11ea-8e51-080bd02a1856.png)


```julia
@parameters t, x, θ
@variables u1(..), u2(..), u3(..)
@derivatives Dt'~t
@derivatives Dtt''~t
@derivatives Dx'~x
@derivatives Dxx''~x

eqs = [Dtt(u1(t,x,θ)) ~ Dxx(u1(t,x,θ)) + u3(t,x,θ)*sin(pi*x),
       Dtt(u2(t,x,θ)) ~ Dxx(u2(t,x,θ)) + u3(t,x,θ)*cos(pi*x),
       0. ~ u1(t,x,θ)*sin(pi*x) + u2(t,x,θ)*cos(pi*x) - exp(-t)]

bcs = [u1(0,x,θ) ~ sin(pi*x),
       u2(0,x,θ) ~ cos(pi*x),
       Dt(u1(0,x,θ)) ~ -sin(pi*x),
       Dt(u2(0,x,θ)) ~ -cos(pi*x),
       u1(t,0,θ) ~ 0.,
       u2(t,0,θ) ~ exp(-t),
       u1(t,1,θ) ~ 0.,
       u2(t,1,θ) ~ -exp(-t),
       u1(t,0,θ) ~ u1(t,1,θ),
       u2(t,0,θ) ~ -u2(t,1,θ)]


# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
# Neural network
input_ = length(domains)
output = length(eqs)
chain = FastChain(FastDense(input_,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,output))

strategy = GridTraining()
discretization = PhysicsInformedNN(dx,chain,strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[u1,u2,u3])
prob = discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=2000)
phi = discretization.phi
```

And some analysis:

```julia
ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

analytic_sol_func(t,x) = [exp(-t)*sin(pi*x), exp(-t)*cos(pi*x), (1+pi^2)*exp(-t)]
u_real  = [[analytic_sol_func(t,x)[i] for t in ts for x in xs] for i in 1:3]
u_predict  = [[phi([t,x],res.minimizer)[i] for t in ts for x in xs] for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]

for i in 1:3
    p1 = plot(xs, ts, u_real[i], st=:surface,title = "u$i, analytic");
    p2 = plot(xs, ts, u_predict[i], st=:surface,title = "predict");
    p3 = plot(xs, ts, diff_u[i],linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("sol_u$i")
end
```

![u1](https://user-images.githubusercontent.com/12683885/90981503-192e9a00-e56a-11ea-8378-8e53e9de9c3c.png)

![u2](https://user-images.githubusercontent.com/12683885/90981470-e1bfed80-e569-11ea-9210-ff606af17532.png)

![u3](https://user-images.githubusercontent.com/12683885/90981491-01571600-e56a-11ea-9143-52ced4b177c8.png)


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
bcs = [u[1](x,0,θ) ~ x, u[2](x,0,θ) ~ 2, u[3](x,0,θ) ~ 3, u[4](x,0,θ) ~ 4]
```

## Example 5 : Solving an ODE with a 3rd-order derivative

Let's consider the ODE with a 3rd-order derivative:

![hdode](https://user-images.githubusercontent.com/12683885/89736407-dc46ab80-da71-11ea-9c6e-5964488642bd.png)

with grid discretization `dx = 0.1`.

```julia
@parameters x θ
@variables u(..)
@derivatives Dxxx'''~x
@derivatives Dx'~x

# ODE
eq = Dxxx(u(x,θ)) ~ cos(pi*x)

# Initial and boundary conditions
bcs = [u(0.,θ) ~ 0.0,
       u(1.,θ) ~ cos(pi),
       Dx(u(1.,θ)) ~ 1.0]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.05

# Neural network
chain = FastChain(FastDense(1,8,Flux.σ),FastDense(8,1))

discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy=StochasticTraining(include_frac=0.5))
pde_system = PDESystem(eq,bcs,domains,[x],[u])
prob = discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.01), progress = false; cb = cb, maxiters=2000)
phi = discretization.phi
```

We can plot the predicted solution of the ODE and its analytical solution.

```julia
analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)

xs = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

x_plot = collect(xs)
plot(x_plot ,u_real,title = "real")
plot!(x_plot ,u_predict,title = "predict")
```
![hodeplot](https://user-images.githubusercontent.com/12683885/90276340-69bc3e00-de6c-11ea-89a7-7d291123a38b.png)

## Example 6 : 2-D Burgers' equation, low-level API

Let's consider the Burgers’ equation:

![burgers](https://user-images.githubusercontent.com/12683885/90985032-b6e19380-e581-11ea-89ee-cdfdc4ecf075.png)

Here is an example of using the low-level API:

```julia
@parameters t, x, θ
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
eq  = Dt(u(t,x,θ)) + u(t,x,θ)*Dx(u(t,x,θ)) - (0.01/pi)*Dxx(u(t,x,θ)) ~ 0

# Initial and boundary conditions
bcs = [u(0,x,θ) ~ -sin(pi*x),
       u(t,-1,θ) ~ 0.,
       u(t,1,θ) ~ 0.]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(-1.0,1.0)]
# Discretization
dx = 0.1
# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

strategy = GridTraining()
discretization = PhysicsInformedNN(dx,chain,strategy=strategy)

indvars = [t,x]
depvars = [u]
dim = length(domains)

expr_pde_loss_function = build_loss_function(eq,indvars,depvars)
expr_bc_loss_functions = [build_loss_function(bc,indvars,depvars) for bc in bcs]

train_sets = generate_training_sets(domains,dx,bcs,indvars,depvars)
train_domain_set, train_bound_set, train_set= train_sets

phi = discretization.phi
autodiff = discretization.autodiff
derivative = discretization.derivative
initθ = discretization.initθ

pde_loss_function = get_loss_function(eval(expr_pde_loss_function),
                                      train_domain_set,
                                      phi,
                                      derivative,
                                      strategy)
bc_loss_function = get_loss_function(eval.(expr_bc_loss_functions),
                                     train_bound_set,
                                     phi,
                                     derivative,
                                     strategy)

function loss_function(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end
f = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=1500)
phi = discretization.phi
```

And some analysis:

```julia
ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
u_predict_contourf = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs] ,length(xs),length(ts))
plot(ts, xs, u_predict_contourf, linetype=:contourf,title = "predict")

u_predict = [[first(phi([t,x],res.minimizer)) for x in xs] for t in ts ]
p1= plot(xs, u_predict[2],title = "t = 0.1");
p2= plot(xs, u_predict[6],title = "t = 0.5");
p3= plot(xs, u_predict[end],title = "t = 1");
plot(p1,p2,p3)
```

![burgers](https://user-images.githubusercontent.com/12683885/90984874-a0870800-e580-11ea-9fd4-af8a4e3c523e.png)


![burgers2](https://user-images.githubusercontent.com/12683885/90984856-8c430b00-e580-11ea-9206-1a88ebd24ca0.png)

## Example 7 :  Kuramoto–Sivashinsky equation

Let's consider the Kuramoto–Sivashinsky equation, which contains a 4th-order derivative:

![KS](https://user-images.githubusercontent.com/12683885/91025423-09fb2b00-e602-11ea-8f5c-61e49e4fb54e.png)

with the initial and boundary conditions:

![bs](https://user-images.githubusercontent.com/12683885/91025570-3fa01400-e602-11ea-8fd7-5b0e250a67a4.png)

```julia
@parameters x, t, θ
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dx2''~x
@derivatives Dx3'''~x
@derivatives Dx4''''~x

α = 1
β = 4
γ = 1
eq = Dt(u(x,t,θ)) + u(x,t,θ)*Dx(u(x,t,θ)) + α*Dx2(u(x,t,θ)) + β*Dx3(u(x,t,θ)) + γ*Dx4(u(x,t,θ)) ~ 0

u_analytic(x,t;z = -x/2+t) = 11 + 15*tanh(z) -15*tanh(z)^2 - 15*tanh(z)^3
du(x,t;z = -x/2+t) = 15/2*(tanh(z) + 1)*(3*tanh(z) - 1)*sech(z)^2

bcs = [u(x,0,θ) ~ u_analytic(x,0),
       u(-10,t,θ) ~ u_analytic(-10,t),
       u(10,t,θ) ~ u_analytic(10,t),
       Dx(u(-10,t,θ)) ~ du(-10,t),
       Dx(u(10,t,θ)) ~ du(10,t)]

# Space and time domains
domains = [x ∈ IntervalDomain(-10.0,10.0),
           t ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.4; dt = 0.2

# Neural network
chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))

discretization = PhysicsInformedNN([dx,dt],
                                    chain,
                                    strategy = GridTraining())
pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = discretize(pde_system,discretization)

opt = Optim.BFGS()
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=2000)
phi = discretization.phi
```

And some analysis:

```julia
xs,ts = [domain.domain.lower:dx:domain.domain.upper for (domain,dx) in zip(domains,[dx/10,dt])]

u_predict = [[first(phi([x,t],res.minimizer)) for x in xs] for t in ts]
u_real = [[u_analytic(x,t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x,t) -first(phi([x,t],res.minimizer)))  for x in xs] for t in ts]

p1 =plot(xs,u_predict,title = "predict")
p2 =plot(xs,u_real,title = "analytic")
p3 =plot(xs,diff_u,title = "error")
plot(p1,p2,p3)
```

![plotks](https://user-images.githubusercontent.com/12683885/91025889-a6253200-e602-11ea-8f61-8e6e2488e025.png)

## Example 8 : Fokker-Planck equation with GPU acceleration

Let's consider the Fokker-Planck equation:

![fke](https://user-images.githubusercontent.com/12683885/91547965-58c00200-e92d-11ea-8d7b-f20ba79ed7c1.png)

which must satisfy the normalization condition:

![nc](https://user-images.githubusercontent.com/12683885/91548028-74c3a380-e92d-11ea-8ee4-ac2a1c780808.png)

with the boundary conditions:

![bc](https://user-images.githubusercontent.com/12683885/91548102-902eae80-e92d-11ea-8956-736a54e9591e.png)

Actually, it is just enough to add `|>gpu` after `chain` and everything will work.

```julia
# the example is taken from this article https://arxiv.org/abs/1910.10503
@parameters x θ
@variables p(..)
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
α = 0.3
β = 0.5
_σ = 0.5
# Discretization
dx = 0.05
# here we use normalization condition: dx*p(x,θ) ~ 1 in order to get a non-zero solution.
eq  = [(α - 3*β*x^2)*p(x,θ) + (α*x - β*x^3)*Dx(p(x,θ)) ~ (_σ^2/2)*Dxx(p(x,θ)),
       dx*p(x,θ) ~ 1.]

# Initial and boundary conditions
bcs = [p(-2.2,θ) ~ 0. ,p(2.2,θ) ~ 0. , p(-2.2,θ) ~ p(2.2,θ)]

# Space and time domains
domains = [x ∈ IntervalDomain(-2.2,2.2)]

# Neural network
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1)) |>gpu

discretization = NeuralPDE.PhysicsInformedNN(dx,
                                             chain,
                                             strategy= NeuralPDE.GridTraining())

pde_system = PDESystem(eq,bcs,domains,[x],[p])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=8000)
phi = discretization.phi
```

And some analysis:

```julia
analytic_sol_func(x) = 28.022*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))

xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

plot(xs ,u_real, label = "analytic")
plot!(xs ,u_predict, label = "predict")
```

![fkplot](https://user-images.githubusercontent.com/12683885/91551079-47c5bf80-e932-11ea-906e-23c84e2846dc.png)
