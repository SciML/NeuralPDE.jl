using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase, DiffEqOperators
println("NNPDE_tests")
using Test, NeuralNetDiffEq

## Example 1, 1D ode
@parameters t θ
@variables u(..)
@derivatives Dt'~t

# 1D ODE and boundary conditions
eq = Dt(u(t,θ)) ~ t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u(t,θ)*(t + ((1+3*(t^2))/(1+t+t^3)))
bcs = [u(0.) ~ 1.0 , u(1.) ~ 1.202]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0)]

# Method of lines discretization
dx = 0.1
order = 1
discretization = PhysicsInformedNN(dx)

# neural network and optimizer
opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))

pde_system = PDESystem(eq,bcs,domains,[t],[u])
prob = discretize(pde_system,discretization)
alg = NeuralNetDiffEq.NNDE(chain,opt,autodiff=false)
phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1000)

analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
ts = [domain.domain.lower:discretization.dxs/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]

@test u_predict ≈ u_real atol = 0.1

t_plot = collect(ts)
plot(t_plot ,u_real)
plot!(t_plot ,u_predict)

## Example 2, 2D Poisson equation du2/dx2 + du2/dy2 = -1
@parameters x y θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

eq  = Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]

domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]

dx = 0.1
order = 2
discretization = MOLFiniteDifference(dx,order)

opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = NeuralNetDiffEq.NNPDEProblem(pde_system,discretization)
alg = NeuralNetDiffEq.NNDE(chain,opt,autodiff=false)
phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=600)

xs,ys = [domain.domain.lower:discretization.dxs:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))

@test u_predict ≈ u_real atol = 0.3

p1 =plot(xs, ys, u_predict, st=:surface);
p2 = plot(xs, ys, u_real, st=:surface);
plot(p1,p2)

## Example , 3D PDE
@parameters x y t θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dtt''~t

eq  = Dt(u(x,y,t,θ)) ~ Dxx(u(x,y,t,θ)) + Dyy(u(x,y,t,θ))

eq  = 1+t*Dt(u(x,y,t,θ)) ~ Dxx(u(x,y,t,θ)) + t^2+6*Dyy(u(x,y,t,θ))

bcs = [u(x,y,0) ~ exp(x+y)*cos(x+y) ,
       u(x,y,2) ~ exp(x+y)*cos(x+y+4*2) ,
       u(0,y,t) ~ exp(y)*cos(y+4t),
       u(2,y,t) ~ exp(2+y)*cos(2+y+4t) ,
       u(x,0,t) ~ exp(x)*cos(x+4t),
       u(x,2,t) ~ exp(x+2)*cos(x+2+4t)]

domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]

dx = 0.4
order = 3
discretization = MOLFiniteDifference(dx,order)

opt = Flux.ADAM(0.1)
# opt = BFGS()
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

pde_system = PDESystem(eq,bcs,domains,[x,y,t],[u])
prob = NeuralNetDiffEq.NNPDEProblem(pde_system,discretization)
alg = NeuralNetDiffEq.NNDE(chain,opt,autodiff=false)
phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1300)

xs,ys,ts = [domain.domain.lower:discretization.dxs:domain.domain.upper for domain in domains]
analytic_sol_func(x,y,t) = exp(x+y)*cos(x+y+4t)

u_real = [reshape([analytic_sol_func(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
u_predict = [reshape([first(phi([x,y,t],res.minimizer)) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
@test u_predict ≈ u_real atol = 70.0
