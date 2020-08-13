using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE

## Example 1, 1D ode
@parameters t θ
@variables u(..)
@derivatives Dt'~t

# 1D ODE
eq = Dt(u(t,θ)) ~ t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u(t,θ)*(t + ((1+3*(t^2))/(1+t+t^3)))

# Initial and boundary conditions
bcs = [u(0.,θ) ~ 1.0 , u(1.,θ) ~ 1.202]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.1
discretization = NeuralPDE.PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))

pde_system = PDESystem(eq,bcs,domains,[t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)
phi,res = solve(prob,alg,verbose=true, maxiters=1000)

analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
ts = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]

@test u_predict ≈ u_real atol = 0.2

# t_plot = collect(ts)
# plot(t_plot ,u_real)
# plot!(t_plot ,u_predict)

## Example 2, 2D Poisson equation
@parameters x y θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq  = Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

# Initial and boundary conditions
bcs = [u(0,y,θ) ~ 0.f0, u(1,y,θ) ~ -sin(pi*1)*sin(pi*y),
       u(x,0,θ) ~ 0.f0, u(x,1,θ) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
discretization = NeuralPDE.PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)
phi,res  = solve(prob,alg,verbose=true, maxiters=600)

xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)
u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))

@test u_predict ≈ u_real atol = 0.3

# p1 =plot(xs, ys, u_predict, st=:surface);
# p2 = plot(xs, ys, u_real, st=:surface);
# plot(p1,p2)

## Example , 3D PDE
@parameters x y t θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dt'~t

# 3D PDE
eq  = Dt(u(x,y,t,θ)) ~ Dxx(u(x,y,t,θ)) + Dyy(u(x,y,t,θ))
# Initial and boundary conditions
bcs = [u(x,y,0,θ) ~ exp(x+y)*cos(x+y) ,
       u(x,y,2,θ) ~ exp(x+y)*cos(x+y+4*2) ,
       u(0,y,t,θ) ~ exp(y)*cos(y+4t),
       u(2,y,t,θ) ~ exp(2+y)*cos(2+y+4t) ,
       u(x,0,t,θ) ~ exp(x)*cos(x+4t),
       u(x,2,t,θ) ~ exp(x+2)*cos(x+2+4t)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]

# Discretization
dx = 0.25
discretization = NeuralPDE.PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.1)
# opt = BFGS()
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

pde_system = PDESystem(eq,bcs,domains,[x,y,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)
phi,res  = solve(prob,alg,verbose=true, maxiters=1000)

xs,ys,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_sol_func(x,y,t) = exp(x+y)*cos(x+y+4t)
u_real = [reshape([analytic_sol_func(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
u_predict = [reshape([first(phi([x,y,t],res.minimizer)) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
@test u_predict ≈ u_real atol = 100.0

# p1 =plot(xs, ys, u_predict, st=:surface);
# p2 = plot(xs, ys, u_real, st=:surface);
# plot(p1,p2)

## Example 4, high-order ode
@parameters x θ
@variables u(..)
@derivatives Dxxx'''~x
@derivatives Dx'~x

# ODE
eq = Dxxx(u(x,θ)) ~ cos(pi*x)

# Initial and boundary conditions
# y(0) =0, y(1) =cos(pi)  ,y'(1)=1
bcs = [u(0.,θ) ~ 0.0,
       u(1.,θ) ~ cos(pi),
       Dx(u(1.,θ)) ~ 1.0]
# bcs = [u(0.,θ) ~ 0.0 , u(1.,θ) ~ 0.0, u(0.5,θ) ~ -0.055]

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
phi,res = solve(prob,alg,verbose=true, maxiters=2000)

analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)

# analytic_sol_func(x) = (0.0909939*x - 0.0909939)*x - 0.0322515*sin(π*x)
xs = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

@test u_predict ≈ u_real atol = 10.0

# x_plot = collect(xs)
# plot(x_plot ,u_real)
# plot!(x_plot ,u_predict)

## Example 5, system of pde
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

analytic_sol_func(x,y) =[1/3*(6x - y), 1/2*(6x - y)]
xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
u_real  = [analytic_sol_func(x,y) for x in xs  for y in ys]
u_predict  = [phi([x,y],res.minimizer) for x in xs  for y in ys]

@test u_predict ≈ u_real atol = 10.0

## Example 6, 2d wave equation, neumann boundary condition
#here we use inner functions for build solution
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
opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
alg = NeuralPDE.NNDE(chain,opt,autodiff=false)

indvars = [x,t]
depvars = [u]
dim = length(domains)

pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars)
bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars) for bc in bcs]
train_sets = NeuralPDE.generate_training_sets(domains,discretization,bcs,indvars,depvars)
loss_function = NeuralPDE.get_loss_function(eval(pde_loss_function),
                                            eval.(bc_loss_functions),
                                            train_sets)
prob = GalacticOptim.OptimizationProblem(loss_function, zeros(dim))

phi,res  = solve(prob,alg,verbose=true, maxiters=1000)

xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_sol_func(x,t) =  sum([(8/(k^3*pi^3)) * sin(k*pi*x)*cos(C*k*pi*t) for k in 1:2:50000])

u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))

@test u_predict ≈ u_real atol = 10.0

# diff_u = abs.(u_predict .- u_real)
# p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
# p2 =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)
