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

## Coordinate Transform and Differential Operators

We represent the field vairable, ``u``, and physical coordinates ``x,y`` in terms of reference variables ``r,\theta``:

```math
u(r,\theta), \, x(r,\theta), \, y(r,\theta)
```

```julia
@paameters r,\theta
@variables u(..)

x = r*cos(\theta)
y = r*sin(\theta)
```

To obtain derivateives with respect to ``x`` and ``y``, we employ the chain rule:

```math
\begin{align}
\partial_x u(r,s) &= u_r(r,s)\partial_x r + u_s(x,y)\partial_x s\\
\partial_y u(r,s) &= u_r(r,s)\partial_y r + u_s(x,y)\partial_y s
\end{align}

\implies
\begin{bmatrix} \partial_x \\ \partial_y \end{bmatrix} u(r,s)
=
\begin{bmatrix} r_x & \theta_x\\ r_y & \theta_y \end{bmatrix}
\begin{bmatrix} \partial_r \\ \partial_s \end{bmatrix} u(r,s)
```

To take gradients with respect to ``x,y``, we need to find ``r_x,\, r_y,\, \theta_x,\, \theta_y``. We begin by observing that
```math
\begin{align}
\partial_x x(r,s) &= 1,\\
\partial_y x(r,s) &= 0,\\
\partial_x y(r,s) &= 0,\\
\partial_y y(r,s) &= 1
\end{align}

\implies
\begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}
=
\begin{bmatrix} r_x & \theta_x\\ r_y & \theta_y \end{bmatrix}
\begin{bmatrix} x_r & y_r\\ x_\theta & y_\theta \end{bmatrix}

\implies
\begin{bmatrix} r_x & \theta_x\\ r_y & \theta_y \end{bmatrix}
=
(1/J)
\begin{bmatrix} y_\theta & -y_r\\ -x_\theta & x_r \end{bmatrix}
```
The gradients are implemented as follows:
```julia
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
```

The second derivaties are obtained by by composing first derivaties:

```julia
Dxx = Dx ∘ Dx
Dyy = Dy ∘ Dy
```

![poisson_annulus_plot](https://user-images.githubusercontent.com/12683885/90962648-2db35980-e4ba-11ea-8e58-f4f07c77bcb9.png)
