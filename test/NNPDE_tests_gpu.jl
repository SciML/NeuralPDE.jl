using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using CUDA

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 1, 3D PDE
@parameters x y t
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dt'~t

# 3D PDE
eq  = Dt(u(x,y,t)) ~ Dxx(u(x,y,t)) + Dyy(u(x,y,t))
# Initial and boundary conditions
bcs = [u(x,y,0) ~ exp(x+y)*cos(x+y) ,
      # u(x,y,2) ~ exp(x+y)*cos(x+y+4*2) ,
      u(0,y,t) ~ exp(y)*cos(y+4t),
      u(2,y,t) ~ exp(2+y)*cos(2+y+4t) ,
      u(x,0,t) ~ exp(x)*cos(x+4t),
      u(x,2,t) ~ exp(x+2)*cos(x+2+4t)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
          y ∈ IntervalDomain(0.0,2.0),
          t ∈ IntervalDomain(0.0,2.0)]

# Discretization
dx = 0.25; dy= 0.25; dt = 0.25
# Neural network
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

initθ = initial_params(chain) |>gpu
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             initθ,
                                             strategy = NeuralPDE.GridTraining(dx=[dx,dy,dt]))
pde_system = PDESystem(eq,bcs,domains,[x,y,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=1000)
phi = discretization.phi

xs,ys,ts = [domain.domain.lower:dx:domain.domain.upper for (dx,domain) in zip([dx,dy,dt],domains)]
analytic_sol_func(x,y,t) = exp(x+y)*cos(x+y+4t)
u_real = [reshape([analytic_sol_func(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
u_predict = [reshape([first(phi([x,y,t],res.minimizer)) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
@test u_predict ≈ u_real atol = 200.0

# p1 =plot(xs, ys, u_predict, st=:surface);
# p2 = plot(xs, ys, u_real, st=:surface);
# plot(p1,p2)

## Example 2, ## Fokker-Planck equation
# the example took from this article https://arxiv.org/abs/1910.10503
@parameters x
@variables p(..)
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
α = 0.3
β = 0.5
_σ = 0.5
# Discretization
dx = 0.05
# here we use normalization condition: dx*p(x) ~ 1, in order to get non-zero solution.
eq  = [(α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x)),
       dx*p(x) ~ 1.]

# Initial and boundary conditions
bcs = [p(-2.2) ~ 0. ,p(2.2) ~ 0. , p(-2.2) ~ p(2.2)]

# Space and time domains
domains = [x ∈ IntervalDomain(-2.2,2.2)]

# Neural network
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))

initθ = initial_params(chain) |>gpu
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             initθ,
                                             strategy= NeuralPDE.GridTraining(dx=dx))

pde_system = PDESystem(eq,bcs,domains,[x],[p])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=1000)
phi = discretization.phi

analytic_sol_func(x) = 28*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))

xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

@test u_predict ≈ u_real atol = 20.0

# plot(xs ,u_real, label = "analytic")
# plot!(xs ,u_predict, label = "predict")
