# Poisson Equation

In this example, we solve the Poisson equation on an annulus to demostrate how NeuralPDE can solve differential equations in complex geometries.

```math
\begin{align}
-(∂^2_x + ∂^2_y)u &= 1 \, (x,y)\in\Omega, \\
\Omega &= \{(x,y) | 0.5 \leq x^2 + y^2 \leq 1.0 \}
\end{align}
```

We represent *physical* coordinates, ``(x,y)``, and field variable ``u`` in terms of reference coordinates ``r,\theta`` which stand for *radius*, and *angle* respectively. We apply the following boundary conditions:

```math
u|_{r=0.5} = u|_{r=1.0} = 0
```

## Copy-Pastable Code

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum
using Quadrature, Cuba, QuasiMonteCarlo
using CUDA, LinearAlgebra

using Plots

using ParameterSchedulers:Scheduler,Sin,SinExp

r0 = 0.5f0 # inner radius
r1 = 1.0f0 # outer radius

@parameters r θ
@variables u(..)

x = r*cos(θ)
y = r*sin(θ)

Dr = Differential(r)
Dθ = Differential(θ)

xr = Dr(x); yr = Dr(y)
xθ = Dθ(x); yθ = Dθ(y)

J  = xr*yθ - xθ*yr
Ji = 1 / J

rx =  Ji * yθ
ry = -Ji * xθ
θx = -Ji * yr
θy =  Ji * xr

Dx(v) = rx*Dr(v) + θx*Dθ(v)
Dy(v) = ry*Dr(v) + θy*Dθ(v)

Dxx = Dx ∘ Dx
Dyy = Dy ∘ Dy

# 2D PDE
eq = -(Dxx(u(r,θ)) + Dyy(u(r,θ))) ~ 1.0f0

# Boundary conditions
bcs = [u(r0, θ) ~ 0.f0       # Dirichlet, inner
      ,u(r1, θ) ~ 0.f0       # Dirichlet, outer
      ,u(r,0f0) ~ u(r,2f0pi) # Periodic
      ]

domains = [r ∈ Interval(r0 ,r1),
           θ ∈ Interval(0f0,2f0pi)]

pdesys = PDESystem(eq, bcs, domains, [r, θ], [u])


# Discretization
ndim  = 2
width = 32
depth = 2
act   = σ

NN = Chain(Dense(ndim,width,act)
          ,Chain([Dense(width,width,act) for i in 1:(depth-1)]...)
          ,Dense(width,1))


initθ    = DiffEqFlux.initial_params(NN)
strategy = QuadratureTraining()
discr    = PhysicsInformedNN(NN,strategy,init_params=initθ)
prob     = discretize(pdesys,discr)

# callback function
iter = []
loss = []
i = 0
ifv = true # verbose flag
function cb(p,l)
    ifv && ((i % 10) == 0) && println("Iter: $i, loss: $l")
    global i += 1
    push!(iter,i)
    push!(loss,l)
    (l < 1e-6) && return true # 5x machine epsilon for float32
    return false
end

# training
maxiters_opt1 = 500
maxiters_opt2 = 300

sch  = SinExp(λ0=1e-3,λ1=1e-1,γ=0.5,period=100)
opt1 = Scheduler(sch,ADAM())
opt2 = BFGS()

res  = GalacticOptim.solve(prob,opt1;cb=cb,maxiters=maxiters_opt1)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,opt2;cb=cb,maxiters=maxiters_opt2)

# solution
phi  = discr.phi
minimizer = res.minimizer

# evaulate solution
ns = 50
rs,θs=[Array(range(infimum(d.domain),supremum(d.domain),length=ns))
                                            for d in pdesys.domain]

o  = ones(ns)
rs = rs * o'
θs = o  * θs'

coord(r,θ) = @. r*cos(θ), r*sin(θ)

xs,ys = coord(rs,θs)

v = zeros(Float32,2,ns*ns)
v[1,:] = rs[:]
v[2,:] = θs[:]

u_predict = phi(v,minimizer)
u_predict = reshape(u_predict,ns,ns)

function meshplt(x,y,u;a=45,b=60)
    p = plot(x,y,u,legend=false,c=:grays,camera=(a,b))
    p = plot!(x',y',u',legend=false,c=:grays,camera=(a,b))
    return p
end

plt = meshplt(xs,ys,u_predict)
```

## Detailed Description

The ModelingToolkit PDE interface for this example looks like this:

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

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
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]
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
xs,ys = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]
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
