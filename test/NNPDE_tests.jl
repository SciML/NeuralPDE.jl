using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 1, 1D ode
println("Example 1, 1D ode")
@parameters θ
@variables u(..)
Dθ = Differential(θ)

# 1D ODE
eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))

# Initial and boundary conditions
bcs = [u(0.) ~ 1.0]

# Space and time domains
domains = [θ ∈ IntervalDomain(0.0,1.0)]
# Discretization
dt = 0.1
# Neural network
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))

strategy = NeuralPDE.QuadratureTraining()
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             )

pde_system = PDESystem(eq,bcs,domains,[θ],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=1000)
prob2 = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob2, ADAM(0.001); cb = cb, maxiters=1000)
phi = discretization.phi

analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
ts = [domain.domain.lower:dt/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]

@test u_predict ≈ u_real atol = 0.2

# using Plots
# t_plot = collect(ts)
# plot(t_plot ,u_real)
# plot!(t_plot ,u_predict)

## Example 2, 2D Poisson equation
function test_2d_poisson_equation(chain_, strategy_)
    println("Example 2, 2D Poisson equation, chain: $(typeof(chain)), strategy: $strategy_")
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    # Initial and boundary conditions
    bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
           u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
    # Space and time domains
    domains = [x ∈ IntervalDomain(0.0,1.0),
               y ∈ IntervalDomain(0.0,1.0)]

    # chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
    discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                 strategy_)

    pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
    prob = NeuralPDE.discretize(pde_system,discretization)

    res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=500)
    phi = discretization.phi

    dx = 0.1
    xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
    analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

    u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
    u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    diff_u = abs.(u_predict .- u_real)

    @test u_predict ≈ u_real atol = 3.0

    # p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    # p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    # p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    # plot(p1,p2,p3)
end

# Discretization
dx = 0.1
grid_strategy = NeuralPDE.GridTraining(dx)
fastchain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
fluxchain = Chain(Dense(2,12,Flux.σ),Dense(12,12,Flux.σ),Dense(12,1))
chains = [fluxchain, fastchain]
for chain in chains
    test_2d_poisson_equation(chain, grid_strategy)
end

stochastic_strategy = NeuralPDE.StochasticTraining(100) #points
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),
                                                   reltol = 1e-2, abstol = 1e-2,
                                                   maxiters = 50)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100; #points
                                                     sampling_alg = UniformSample(),
                                                     minibatch = 100)

strategies = [stochastic_strategy, quadrature_strategy,quasirandom_strategy]
for strategy_ in strategies
    chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
    test_2d_poisson_equation(chain_, strategy_)
end

function run_2d_poisson_equation(strategy_)
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    # Initial and boundary conditions
    bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
           u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
    # Space and time domains
    domains = [x ∈ IntervalDomain(0.0,1.0),
               y ∈ IntervalDomain(0.0,1.0)]
    # Discretization
    dx = 0.1

    chain = FastChain(FastDense(2,4,Flux.σ),FastDense(4,1))

    discretization = NeuralPDE.PhysicsInformedNN(chain,
                                                 strategy_)

    pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
    prob = NeuralPDE.discretize(pde_system,discretization)

    res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb,  maxiters=5)
end

algs = [HCubatureJL(), CubatureJLh(), CubatureJLp(),CubaCuhre()]
for alg in algs
    strategy_ =  NeuralPDE.QuadratureTraining(quadrature_alg = alg,reltol=1e-8,abstol=1e-8,maxiters=30)
    run_2d_poisson_equation(strategy_)
end


## Example 3, high-order ode
println("Example 3, high-order ode")
@parameters x
@variables u(..)
Dxxx = Differential(x)^3
Dx = Differential(x)

# ODE
eq = Dxxx(u(x)) ~ cos(pi*x)

# Initial and boundary conditions
bcs = [u(0.) ~ 0.0,
       u(1.) ~ cos(pi),
       Dx(u(1.)) ~ 1.0]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.05

# Neural network
chain = FastChain(FastDense(1,8,Flux.σ),FastDense(8,1))


discretization = NeuralPDE.PhysicsInformedNN(chain,NeuralPDE.GridTraining(dx))
pde_system = PDESystem(eq,bcs,domains,[x],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=200)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=800)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob, ADAM(0.001); cb = cb, maxiters=400)
phi = discretization.phi

analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)

xs = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

@test u_predict ≈ u_real atol = 1.0

# x_plot = collect(xs)
# plot(x_plot ,u_real)
# plot!(x_plot ,u_predict)

## Example 4, system of pde
println("Example 4, system of pde")
@parameters x, y
@variables u1(..), u2(..)
Dx = Differential(x)
Dy = Differential(y)

# System of pde
eqs = [Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
      Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0,
      3*u1(x,0) ~ 2*u2(x,0)]

# Initial and boundary conditions
bcs = [u1(x,0) ~ 2x, u2(x,0) ~ 3x]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0), y ∈ IntervalDomain(0.0,1.0)]


# Neural network
chain1 = FastChain(FastDense(2,10,Flux.σ),FastDense(10,1))
chain2 = FastChain(FastDense(2,10,Flux.σ),FastDense(10,1))
strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubaCuhre(),reltol= 1e-4,abstol= 1e-3,maxiters=20)
discretization = NeuralPDE.PhysicsInformedNN([chain1,chain2],strategy)
pde_system = PDESystem(eqs,bcs,domains,[x,y],[u1,u2])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=300)
phi = discretization.phi

analytic_sol_func(x,y) =[1/3*(6x - y), 1/2*(6x - y)]
xs,ys = [domain.domain.lower:0.01:domain.domain.upper for domain in domains]
u_real  = [[analytic_sol_func(x,y)[i] for x in xs  for y in ys] for i in 1:2]

initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
u_predict  = [[phi[i]([x,y],minimizers[i])[1] for x in xs  for y in ys] for i in 1:2]

@test u_predict ≈ u_real atol = 2.0

# p1 =plot(xs, ys, u_predict, st=:surface);
# p2 = plot(xs, ys, u_real, st=:surface);
# plot(p1,p2)

## Example 5, 2d wave equation, neumann boundary condition
println("Example 5, 2d wave equation, neumann boundary condition")
#here we use low level api for build solution
@parameters x, t
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(x,t)) ~ C^2*Dxx(u(x,t))

# Initial and boundary conditions
bcs = [u(0,t) ~ 0.,# for all t > 0
       u(1,t) ~ 0.,# for all t > 0
       u(x,0) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(x,0)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

phi = NeuralPDE.get_phi(chain)
derivative = NeuralPDE.get_numeric_derivative()
initθ = DiffEqFlux.initial_params(chain)

indvars = [x,t]
depvars = [u]
dim = length(domains)

_pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,phi, derivative,initθ)

bc_indvars = NeuralPDE.get_bc_varibles(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars, phi, derivative,initθ,
                                              bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

dx = 0.1
train_sets = NeuralPDE.generate_training_sets(domains,dx,bcs,indvars,depvars)
pde_train_set,bcs_train_set,train_set = train_sets
pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,bcs,indvars,depvars)

quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),reltol= 1e-4,abstol= 1e-3,maxiters=10)
pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                                                pde_bounds,
                                                quadrature_strategy;
                                                τ = 1/100)


quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),reltol= 1e-2,abstol= 1e-1,maxiters=5)
bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                                               bcs_bounds,
                                               quadrature_strategy;
                                               τ = 1/40)

function loss_function_(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end

f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

cb_ = function (p,l)
    println("Current losses are: ", pde_loss_function(p), " , ",  bc_loss_function(p))
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb_, maxiters=400)

xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_sol_func(x,t) =  sum([(8/(k^3*pi^3)) * sin(k*pi*x)*cos(C*k*pi*t) for k in 1:2:50000])

u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))

@test u_predict ≈ u_real atol = 1.0

# diff_u = abs.(u_predict .- u_real)
# p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
# p2 =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)


## Example 6, pde with mixed derivative
println("Example 6, pde with mixed derivative")
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dx = Differential(x)
Dy = Differential(y)

eq = Dxx(u(x,y)) + Dx(Dy(u(x,y))) - 2*Dyy(u(x,y)) ~  -1.

# Initial and boundary conditions
bcs = [u(x,0) ~ x,
       Dy(u(x,0)) ~ x,
       u(x,0) ~ Dy(u(x,0))]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0), y ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.1
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),reltol= 1e-2,abstol= 1e-2,maxiters=10)
# Neural network
chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
discretization = NeuralPDE.PhysicsInformedNN(chain, quadrature_strategy)
pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=200)
phi = discretization.phi

analytic_sol_func(x,y) = x + x*y +y^2/2
xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real atol = 1.0

# p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
# p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)

## Example 7, Fokker-Planck equation
println("Example 7, Fokker-Planck equation")
# the example took from this article https://arxiv.org/abs/1910.10503
@parameters x
@variables p(..)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
α = 0.3
β = 0.5
_σ = 0.5
# Discretization
dx = 0.05
# here we use normalization condition: dx*p(x) ~ 1, in order to get non-zero solution.
#(α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x))
eq  = Dx((α*x - β*x^3)*p(x)) ~ (_σ^2/2)*Dxx(p(x))+dx*p(x) - 1.

# Initial and boundary conditions
bcs = [p(-2.2) ~ 0. ,p(2.2) ~ 0. , p(-2.2) ~ p(2.2)]

# Space and time domains
domains = [x ∈ IntervalDomain(-2.2,2.2)]

# Neural network
chain = FastChain(FastDense(1,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dx))

pde_system = PDESystem(eq,bcs,domains,[x],[p])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=800)

pde_system2 = remake(pde_system; eqs = Dx((α*x - β*x^3)*p(x)) ~ (_σ^2/2)*Dxx(p(x)))
discretization2 = remake(discretization; strategy = NeuralPDE.GridTraining(dx/5), init_params =res.minimizer)
prob2 = NeuralPDE.discretize(pde_system2,discretization2)

res = GalacticOptim.solve(prob2,Optim.BFGS();cb=cb,maxiters=200)
phi = discretization.phi

analytic_sol_func(x) = 28*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))

xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,Array(res.minimizer))) for x in xs]

@test u_predict ≈ u_real atol = 6.0

# plot(xs ,u_real, label = "analytic")
# plot!(xs ,u_predict, label = "predict")
