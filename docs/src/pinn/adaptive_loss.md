# Adaptive Losses and SPM

In this example, we will use adaptive losses to solve the Single Particle Model (SPM), a simplified battery chemistry model:

```math
∂^2_x u(x, y) + ∂^2_y u(x, y) = - \sin(\pi x) \sin(\pi y) \, ,
```

with the boundary conditions:

```math
\begin{align*}
u(0, y) &= 0 \, ,\\
u(1, y) &= - \sin(\pi) \sin(\pi y) \, ,\\
u(x, 0) &= 0 \, ,\\
u(x, 1) &=  - \sin(\pi x) \sin(\pi) \, ,
\end{align*}
```

on the space domain:

```math
x \in [0, 1] \, , \ y \in [0, 1] \, ,
```

with grid discretization `dx = 0.1`. We will use physics-informed neural networks.

## Copy-Pastable Code

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

# Boundary conditions
bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
# Initial parameters of Neural network
initθ = Float64.(DiffEqFlux.initial_params(chain))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain,GridTraining(dx),init_params =initθ)

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
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

using Plots

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
bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
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

Convert weights of neural network from Float32 to Float64 in order to all inner calculation will be with Float64.

```julia
# Initial parameters of Neural network
initθ = Float64.(DiffEqFlux.initial_params(chain))
```

Here, we build PhysicsInformedNN algorithm where `dx` is the step of discretization, `strategy` stores information for choosing a training strategy and
`init_params =initθ` initial parameters of neural network.

```julia
# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx),init_params =initθ)
```

As described in the API docs, we now need to define the `PDESystem` and create PINNs problem using the `discretize` method.

```julia
@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
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

```julia
using LinearAlgebra
#using PyCall
using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using DelimitedFiles
using CSV
using Plots

using Random

seed!(100)
# ('negative particle',) -> rn
# ('positive particle',) -> rp
@parameters t rn rp
# 'Discharge capacity [A.h]' -> Q
# 'X-averaged negative particle concentration' -> c_s_n_xav
# 'X-averaged positive particle concentration' -> c_s_p_xav
@variables Q(..) c_s_n_xav(..) c_s_p_xav(..)
Dt = Differential(t)
Drn = Differential(rn)
Drp = Differential(rp)

eqs = [
Dt(Q(t)) ~ 4.27249308415467,
# 'X-averaged negative particle concentration' equation
Dt(c_s_n_xav(t, rn)) ~ 8.813457647415216 * (Drn(Drn(c_s_n_xav(t, rn))) + 2 / rn * Drn(c_s_n_xav(t, rn))),
# 'X-averaged positive particle concentration' equation
Dt(c_s_p_xav(t, rp)) ~ 22.598609352346717 * (Drp(Drp(c_s_p_xav(t, rp))) + 2 / rp * Drp(c_s_p_xav(t, rp))),
]

ics_bcs = [
Q(0) ~ 0.0,
c_s_n_xav(0, rn) ~ 0.8,
c_s_p_xav(0, rp) ~ 0.6,
Drn(c_s_n_xav(t, 0.01)) ~ 0.0,
Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
Drp(c_s_p_xav(t, 0.01)) ~ 0.0,
Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
]

t_domain = IntervalDomain(0.0, 0.15) 
rn_domain = IntervalDomain(0.01, 1.0)
rp_domain = IntervalDomain(0.01, 1.0)

domains = [
t in t_domain,
rn in rn_domain,
rp in rp_domain,
]
ind_vars = [t, rn, rp]
dep_vars = [Q(t), c_s_n_xav(t, rn), c_s_p_xav(t, rp)]

SPM_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)


num_dim = 50
nonlin = Flux.gelu
strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),abstol=1e-4, reltol=1, maxiters=2000, batch=0)
in_dims = [1, 2, 2]
num_hid = 2
chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                    [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                    FastDense(num_dim,1)) for in_dim in in_dims]
adaloss = NeuralPDE.MiniMaxAdaptiveLoss(20; pde_max_optimiser=ADAM(1e-1), bc_max_optimiser=ADAM(1e1), pde_loss_weights=1e-3, bc_loss_weights=1e3)
discretization = NeuralPDE.PhysicsInformedNN(chains_, strategy; adaptive_loss=adaloss)
sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
prob = NeuralPDE.discretize(SPM_pde_system,discretization)

initθ = vcat(discretization.init_params...)
opt = ADAM(3e-4)
```
