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

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 1, 1D ode
@parameters θ
@variables u(..)
@derivatives Dθ'~θ

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


discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             nothing; #init_params
                                             phi = nothing,
                                             autodiff=false,
                                             derivative = nothing,
                                             strategy = NeuralPDE.GridTraining(dx=dt))

pde_system = PDESystem(eq,bcs,domains,[θ],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=1000)
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
function test_2d_poisson_equation(chain, strategy)
    @parameters x y
    @variables u(..)
    @derivatives Dxx''~x
    @derivatives Dyy''~y

    # 2D PDE
    eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    # Initial and boundary conditions
    bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
           u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
    # Space and time domains
    domains = [x ∈ IntervalDomain(0.0,1.0),
               y ∈ IntervalDomain(0.0,1.0)]

    # chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
    discretization = NeuralPDE.PhysicsInformedNN(chain,
                                                 strategy = strategy)

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
grid_strategy = NeuralPDE.GridTraining(dx=dx)
fastchain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
fluxchain = Chain(Dense(2,12,Flux.σ),Dense(12,12,Flux.σ),Dense(12,1))
chains = [fluxchain, fastchain]
for chain in chains
    test_2d_poisson_equation(chain, grid_strategy)
end

stochastic_strategy = NeuralPDE.StochasticTraining(number_of_points = 100)
quadrature_strategy = NeuralPDE.QuadratureTraining(algorithm=HCubatureJL(),
                                                   reltol = 1e-2, abstol = 1e-2,
                                                   maxiters = 50)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(sampling_method = UniformSample(),
                                                     number_of_points = 100,
                                                     number_of_minibatch = 10)

strategies = [stochastic_strategy, quadrature_strategy,quasirandom_strategy]
for strategy in strategies
    chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
    test_2d_poisson_equation(chain, strategy)
end

function run_2d_poisson_equation(strategy)
    @parameters x y
    @variables u(..)
    @derivatives Dxx''~x
    @derivatives Dyy''~y

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
                                                 strategy = strategy)

    pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
    prob = NeuralPDE.discretize(pde_system,discretization)

    res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb,  maxiters=2)
end

algs = [CubaVegas(), CubaSUAVE(),HCubatureJL(), CubatureJLh(), CubatureJLp()]
#CubaDivonne(),CubaCuhre() doesn't work with dim = 2
for alg in algs
    strategy =  NeuralPDE.QuadratureTraining(algorithm = alg,reltol=1e-8,abstol=1e-8,maxiters=600)
    run_2d_poisson_equation(strategy)
end


## Example 3, high-order ode
@parameters x
@variables u(..)
@derivatives Dxxx'''~x
@derivatives Dx'~x

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


discretization = NeuralPDE.PhysicsInformedNN(chain,strategy = NeuralPDE.GridTraining(dx=dx))
pde_system = PDESystem(eq,bcs,domains,[x],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=2000)
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
@parameters x, y
@variables u1(..), u2(..)
@derivatives Dx'~x
@derivatives Dy'~y

# System of pde
eqs = [Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
      Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0,
      3*u1(x,0) ~ 2*u2(x,0)]

# Initial and boundary conditions
bcs = [u1(x,0) ~ 2x, u2(x,0) ~ 3x]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0), y ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.1

# Neural network
chain1 = FastChain(FastDense(2,8,Flux.σ),FastDense(8,1))
chain2 = FastChain(FastDense(2,8,Flux.σ),FastDense(8,1))
quadrature_strategy = NeuralPDE.QuadratureTraining(algorithm=HCubatureJL(),reltol= 1e-4,abstol= 1e-4,maxiters=1e2)
discretization = NeuralPDE.PhysicsInformedNN([chain1,chain2],strategy = quadrature_strategy)
pde_system = PDESystem(eqs,bcs,domains,[x,y],[u1,u2])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=300)
phi = discretization.phi

analytic_sol_func(x,y) =[1/3*(6x - y), 1/2*(6x - y)]
xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
u_real  = [[analytic_sol_func(x,y)[i] for x in xs  for y in ys] for i in 1:2]

initθ = discretization.initθ
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
u_predict  = [[phi[i]([x,y],minimizers[i])[1] for x in xs  for y in ys] for i in 1:2]

@test u_predict ≈ u_real atol = 10.0

# p1 =plot(xs, ys, u_predict, st=:surface);
# p2 = plot(xs, ys, u_real, st=:surface);
# plot(p1,p2)

## Example 5, 2d wave equation, neumann boundary condition
#here we use low level api for build solution
@parameters x, t
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t

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
# Discretization
dx = 0.1

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

train_sets = NeuralPDE.generate_training_sets(domains,dx,bcs,indvars,depvars)
pde_train_set,bcs_train_set,train_set = train_sets

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,bcs,indvars,depvars)
grid_strategy = NeuralPDE.GridTraining(dx=dx)
quadrature_strategy = NeuralPDE.QuadratureTraining(algorithm=HCubatureJL(),reltol= 1e-3,abstol= 1e-3,maxiters=20)
pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                                                pde_bounds,
                                                quadrature_strategy)

bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                                               bcs_bounds,
                                               quadrature_strategy)

function loss_function_(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end

f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=400)

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


## Example 6, pde with mixed derivative
@parameters x y
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dy'~y
@derivatives Dx'~x

eq = Dxx(u(x,y)) + Dx(Dy(u(x,y))) - 2*Dyy(u(x,y)) ~  -1.

# Initial and boundary conditions
bcs = [u(x,0) ~ x,
       Dy(u(x,0)) ~ x,
       u(x,0) ~ Dy(u(x,0))]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0), y ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.1
quadrature_strategy = NeuralPDE.QuadratureTraining(algorithm=HCubatureJL(),reltol= 1e-2,abstol= 1e-2,maxiters=10)
# Neural network
chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy = quadrature_strategy)
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
