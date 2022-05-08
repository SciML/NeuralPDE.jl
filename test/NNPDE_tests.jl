using Flux, GalacticFlux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
using GalacticOptim
using GalacticOptimJL
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using OrdinaryDiffEq
import ModelingToolkit: Interval, infimum, supremum
using DomainSets

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 1, 1D ode
function test_ode(strategy_)
    println("Example 1, 1D ode: strategy: $(nameof(typeof(strategy_)))")
    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.) ~ 1.0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0,1.0)]

    # Neural network
    chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))
    initθ = Float64.(DiffEqFlux.initial_params(chain))

    discretization = NeuralPDE.PhysicsInformedNN(chain,
                                                 strategy_;
                                                 init_params = initθ,
                                                 phi = nothing,
                                                 derivative = nothing,
                                                 )

    @named pde_system = PDESystem(eq,bcs,domains,[θ],[u])
    prob = NeuralPDE.discretize(pde_system,discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

    res = GalacticOptim.solve(prob, ADAM(0.1); maxiters=1000)
    prob = remake(prob,u0=res.minimizer)
    res = GalacticOptim.solve(prob, ADAM(0.01); maxiters=500)
    prob = remake(prob,u0=res.minimizer)
    res = GalacticOptim.solve(prob, ADAM(0.001); maxiters=500)
    phi = discretization.phi

    analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real  = [analytic_sol_func(t) for t in ts]
    u_predict  = [first(phi(t,res.minimizer)) for t in ts]

    @test u_predict ≈ u_real atol = 0.1
    # using Plots
    # t_plot = collect(ts)
    # plot(t_plot ,u_real)
    # plot!(t_plot ,u_predict)
end

#TODO There is little meaning in these tests without checking the correctness of the prediction.
#TODO I propose to simply remove them.
function test_heterogeneous_equation(strategy_)
	println("Simple Heterogeneous input PDE, strategy: $(nameof(typeof(strategy_)))")
	@parameters x y
	@variables p(..) q(..) r(..) s(..)
	Dx = Differential(x)
	Dy = Differential(y)

	# 2D PDE
	eq  = p(x) + q(y) + Dx(r(x, y)) + Dy(s(y, x)) ~ 0
	# eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

	# Initial and boundary conditions
	bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
			r(x, -1) ~ 0.f0, r(1, y) ~ 0.0f0, 
			s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
	# bcs = [s(y, 1) ~ 0.0f0]
	# Space and time domains
	domains = [x ∈ Interval(0.0, 1.0),
				y ∈ Interval(0.0, 1.0)]

	# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
	numhid = 3
	fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
				[FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
	discretization = NeuralPDE.PhysicsInformedNN(fastchains,
													strategy_)

	@named pde_system = PDESystem(eq, bcs, domains, [x,y], [p(x), q(y), r(x, y), s(y, x)])
	prob = SciMLBase.discretize(pde_system, discretization)
	res = GalacticOptim.solve(prob, BFGS(); maxiters=100)
end

## Heterogeneous system
function test_heterogeneous_system(strategy_)
	println("Heterogeneous input PDE with derivatives, strategy: $(nameof(typeof(strategy_)))")
	@parameters x y
	@variables p(..) q(..)
	Dx = Differential(x)
	Dy = Differential(y)

	# 2D PDE
	#TODO Dx(q(y)) = 0
	#TODO so p(x) = 0, q = const is has only trivial solution
	eq = p(x) + Dx(q(y)) ~ 0

	# Initial and boundary conditions
	bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0]

	# Space and time domains
	domains = [x ∈ Interval(0.0, 1.0),
				y ∈ Interval(-1.0, 0.0)]

	# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
	numhid = 3
	fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
				[FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
	discretization = NeuralPDE.PhysicsInformedNN(fastchains,
													strategy_)

	@named pde_system = PDESystem(eq, bcs, domains, [x,y], [p(x), q(y)])
	prob = SciMLBase.discretize(pde_system, discretization)
	res = GalacticOptim.solve(prob, BFGS(); maxiters=100)
end

grid_strategy = NeuralPDE.GridTraining(0.1)
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =50, batch=100)
stochastic_strategy = NeuralPDE.StochasticTraining(100; bcs_points= 50)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100;
                                                     sampling_alg = LatinHypercubeSample(),
                                                     resampling =false,
                                                     minibatch = 100
                                                    )
quasirandom_strategy_resampling = NeuralPDE.QuasiRandomTraining(100;
                                                     bcs_points= 50,
                                                     sampling_alg = LatticeRuleSample(),
                                                     resampling = true,
                                                     minibatch = 0)

strategies = [grid_strategy,stochastic_strategy, quadrature_strategy,quasirandom_strategy,quasirandom_strategy_resampling]

map(strategies) do strategy_
    test_ode(strategy_)
end
# map(strategies) do strategy_
#     test_heterogeneous_system(strategy_)
# end
# map(strategies) do strategy_
#     test_heterogeneous_equation(strategy_)
# end

## Heterogeneous system
println("Heterogeneous system")

@parameters x,y,z
@variables u(..), v(..), h(..), p(..)
Dz = Differential(z)
eqs = [
    u(x,y,z) ~ x+y+z,
    v(y,x) ~ x^2 + y^2,
    h(z) ~ cos(z),
    p(x,z) ~ exp(x)*exp(z),
    u(x,y,z) + v(y,x)*Dz(h(z)) - p(x,z) ~ x+y+z - (x^2+y^2)*sin(z) - exp(x)*exp(z)
]

bcs = [u(0,0,0) ~ 0.0]

domains = [x ∈ Interval(0.0, 1.0),
           y ∈ Interval(0.0, 1.0),
           z ∈ Interval(0.0, 1.0)]

chain = [FastChain(FastDense(3,12,Flux.tanh),FastDense(12,12,Flux.tanh),FastDense(12,1)),
         FastChain(FastDense(2,12,Flux.tanh),FastDense(12,12,Flux.tanh),FastDense(12,1)),
         FastChain(FastDense(1,12,Flux.tanh),FastDense(12,12,Flux.tanh),FastDense(12,1)),
         FastChain(FastDense(2,12,Flux.tanh),FastDense(12,12,Flux.tanh),FastDense(12,1))]

initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))


grid_strategy = NeuralPDE.GridTraining(0.1)
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =50, batch=100)

discretization = NeuralPDE.PhysicsInformedNN(chain,grid_strategy;init_params = initθ)

@named pde_system = PDESystem(eqs,bcs,domains,[x,y,z],[u(x,y,z),v(y,x),h(z),p(x,z)])

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, BFGS();maxiters=1000)
phi = discretization.phi

analytic_sol_func_ =
[
(x,y,z) -> x+y+z ,
(x,y) -> x^2 + y^2,
(z) -> cos(z),
(x,z) -> exp(x)*exp(z)
]

xs,ys,zs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

u_real = [analytic_sol_func_[1](x,y,z) for x in xs  for y in ys for z in zs]
v_real = [analytic_sol_func_[2](y,x) for y in ys for x in xs ]
h_real = [analytic_sol_func_[3](z) for z in zs]
p_real = [analytic_sol_func_[4](x,z) for x in xs for z in zs]

real_ = [u_real,v_real,h_real,p_real]

initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]


u_predict = [phi[1]([x,y,z],minimizers[1])[1] for x in xs  for y in ys for z in zs]
v_predict = [phi[2]([y,x],minimizers[2])[1] for y in ys for x in xs ]
h_predict = [phi[3]([z],minimizers[3])[1] for z in zs]
p_predict = [phi[4]([x,z],minimizers[4])[1] for x in xs for z in zs]
predict = [u_predict,v_predict,h_predict,p_predict]

for i in 1:4
    @test predict[i] ≈ real_[i] rtol = 10^-3
end

# x_plot = collect(xs)
# y_plot = collect(ys)
# i=1
# z=0
# u_real = collect(analytic_sol_func_[1](x,y,z) for y in ys, x in xs);
# u_predict = collect(phi[1]([x,y,z],minimizers[1])[1]  for y in ys, x in xs);
# plot(x_plot,y_plot,u_real)
# plot!(x_plot,y_plot,u_predict)


## Example 2, 2D Poisson equation
function test_2d_poisson_equation(chain_, strategy_)
    println("Example 2, 2D Poisson equation, chain: $(nameof(typeof(chain_))), strategy: $(nameof(typeof(strategy_)))")
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    # Initial and boundary conditions
    bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
           u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
    # Space and time domains
    domains = [x ∈ Interval(0.0,1.0),
               y ∈ Interval(0.0,1.0)]

    initθ = Float64.(DiffEqFlux.initial_params(chain_))
    discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                 strategy_;
                                                 init_params = initθ)

    @named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
    prob = NeuralPDE.discretize(pde_system,discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
    res = GalacticOptim.solve(prob, ADAM(0.1); maxiters=500)
    phi = discretization.phi

    xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

    u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
    u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    diff_u = abs.(u_predict .- u_real)

    @test u_predict ≈ u_real atol = 2.0

    # p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    # p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    # p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    # plot(p1,p2,p3)
end

fastchain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
fluxchain = Chain(Dense(2,12,Flux.σ),Dense(12,12,Flux.σ),Dense(12,1))
chains = [fluxchain, fastchain]
for chain in chains
    test_2d_poisson_equation(chain, grid_strategy)
end

for strategy_ in strategies
    chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
    test_2d_poisson_equation(chain_, strategy_)
end

algs = [CubatureJLp()] #CubatureJLh(),
for alg in algs
    chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
    strategy_ =  NeuralPDE.QuadratureTraining(quadrature_alg = alg,reltol=1e-4,abstol=1e-3,maxiters=30, batch=10)
    test_2d_poisson_equation(chain_, strategy_)
end


## Example 3, 3rd-order
println("Example 3, 3rd-order ode")
@parameters x
@variables u(..) ,Dxu(..) ,Dxxu(..) ,O1(..), O2(..)
Dxxx = Differential(x)^3
Dx = Differential(x)

# ODE
eq = Dx(Dxxu(x)) ~ cos(pi*x)

# Initial and boundary conditions
bcs_ = [u(0.) ~ 0.0,
        u(1.) ~ cos(pi),
        Dxu(1.) ~ 1.0]
ep = (cbrt(eps(eltype(Float64))))^2/6

der = [Dxu(x) ~ Dx(u(x)) + ep*O1(x) ,
       Dxxu(x) ~  Dx(Dxu(x))+ ep*O2(x)]

bcs = [bcs_;der]
# Space and time domains
domains = [x ∈ Interval(0.0,1.0)]

# Neural network
chain = [[FastChain(FastDense(1,12,Flux.tanh),FastDense(12,12,Flux.tanh),FastDense(12,1)) for _ in 1:3];
         [FastChain(FastDense(1,4,Flux.tanh),FastDense(4,1)) for _ in 1:2];]
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100; #points
                                                     sampling_alg = LatinHypercubeSample())
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain,quasirandom_strategy;init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[x],[u(x),Dxu(x),Dxxu(x),O1(x),O2(x)])

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
discretized_functions = NeuralPDE.discretize_inner_functions(pde_system,discretization)

pde_inner_loss_functions = discretized_functions.pde_loss_functions
bcs_inner_loss_functions = discretized_functions.bc_loss_functions

cb_ = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = GalacticOptim.solve(prob, BFGS(); maxiters=1000)
phi = discretization.phi[1]

analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)

xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

@test u_predict ≈ u_real atol = 10^-4

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
      Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]
      # 3*u1(x,0) ~ 2*u2(x,0)]

# Initial and boundary conditions
bcs = [u1(x,0) ~ 2*x, u2(x,0) ~ 3*x]

# Space and time domains
domains = [x ∈ Interval(0.0,1.0), y ∈ Interval(0.0,1.0)]


# Neural network
chain1 = FastChain(FastDense(2,15,Flux.tanh),FastDense(15,1))
chain2 = FastChain(FastDense(2,15,Flux.tanh),FastDense(15,1))

quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =50, batch=100)
chain = [chain1,chain2]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain,quadrature_strategy; init_params = initθ)

@named pde_system = PDESystem(eqs,bcs,domains,[x,y],[u1(x, y),u2(x, y)])

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,BFGS(); maxiters=1000)
phi = discretization.phi

analytic_sol_func(x,y) =[1/3*(6x - y), 1/2*(6x - y)]
xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_real  = [[analytic_sol_func(x,y)[i] for x in xs  for y in ys] for i in 1:2]

initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
u_predict  = [[phi[i]([x,y],minimizers[i])[1] for x in xs  for y in ys] for i in 1:2]

@test u_predict[1] ≈ u_real[1] atol = 0.1
@test u_predict[2] ≈ u_real[2] atol = 0.1

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
domains = [x ∈ Interval(0.0,1.0),
           t ∈ Interval(0.0,1.0)]

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
eltypeθ = eltype(initθ)
parameterless_type_θ = DiffEqBase.parameterless_type(initθ)
phi = NeuralPDE.get_phi(chain,parameterless_type_θ)
derivative = NeuralPDE.get_numeric_derivative()

indvars = [x,t]
depvars = [u(x, t)]
dim = length(domains)

quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =50, batch=100)

integral = NeuralPDE.get_numeric_integral(quadrature_strategy, indvars, depvars, chain, derivative)
_pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,integral,
                                                   chain,initθ,quadrature_strategy)
_pde_loss_function(rand(2,10), initθ)

bc_indvars = NeuralPDE.get_argument(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars, phi, derivative,integral,
                                                    chain,initθ,quadrature_strategy,
                                                    bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
map(loss_f -> loss_f(rand(1,10), initθ),_bc_loss_functions)

dx = 0.1
train_sets = NeuralPDE.generate_training_sets(domains,dx,[eq],bcs,eltypeθ,indvars,depvars)
pde_train_set,bcs_train_set = train_sets
pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,[eq],bcs,eltypeθ,indvars,depvars,quadrature_strategy)

lbs,ubs = pde_bounds
pde_loss_functions = [NeuralPDE.get_loss_function(_pde_loss_function,
                                                lbs[1],ubs[1],
                                                eltypeθ, parameterless_type_θ,
                                                quadrature_strategy)]

pde_loss_functions[1](initθ)

lbs,ubs = bcs_bounds
bc_loss_functions = [NeuralPDE.get_loss_function(_loss,lb,ub,
                                                 eltypeθ, parameterless_type_θ,
                                                 quadrature_strategy)
                                                 for (_loss,lb,ub) in zip(_bc_loss_functions, lbs,ubs)]

map(l->l(initθ) ,bc_loss_functions)

loss_functions =  [pde_loss_functions;bc_loss_functions]

function loss_function(θ,p)
    sum(map(l->l(θ) ,loss_functions))
end

f_ = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

cb_ = function (p,l)
    println("loss: ", l )
    println("losses: ", map(l -> l(p), loss_functions))
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS(); maxiters=500,f_abstol=10^-6)

xs,ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
analytic_sol_func(x,t) =  sum([(8/(k^3*pi^3)) * sin(k*pi*x)*cos(C*k*pi*t) for k in 1:2:50000])

u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))

@test u_predict ≈ u_real atol = 0.1

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
domains = [x ∈ Interval(0.0,1.0), y ∈ Interval(0.0,1.0)]

quadrature_strategy = NeuralPDE.QuadratureTraining()
# Neural network
inner =20
chain = FastChain(FastDense(2,inner,Flux.tanh),FastDense(inner,inner,Flux.tanh),FastDense(inner,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain, quadrature_strategy; init_params = initθ)
@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,BFGS(); maxiters=1500)
@show res.original

phi = discretization.phi

analytic_sol_func(x,y) = x + x*y +y^2/2
xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real rtol = 0.1

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
dx = 0.01
# here we use normalization condition: dx*p(x) ~ 1, in order to get non-zero solution.
#(α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x))
eq  = [Dx((α*x - β*x^3)*p(x)) ~ (_σ^2/2)*Dxx(p(x))]
x_0 = -2.2
x_end = 2.2
# Initial and boundary conditions
bcs = [p(x_0) ~ 0. ,p(x_end) ~ 0.]

# Space and time domains
domains = [x ∈ Interval(-2.2,2.2)]

# Neural network
inn = 18
chain = FastChain(FastDense(1,inn,Flux.σ),
                  FastDense(inn,inn,Flux.σ),
                  FastDense(inn,inn,Flux.σ),
                  FastDense(inn,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

lb = [x_0]
ub = [x_end]
function norm_loss_function(phi,θ,p)
    function inner_f(x,θ)
        dx*phi(x, θ) .- 1
    end
    prob = QuadratureProblem(inner_f, lb, ub, θ)
    norm2 = solve(prob, HCubatureJL(), reltol = 1e-8, abstol = 1e-8, maxiters =10);
    abs(norm2[1])
end
# norm_loss_function(phi,initθ,nothing)

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dx);
                                             init_params = initθ,
                                             additional_loss=norm_loss_function)

@named pde_system = PDESystem(eq,bcs,domains,[x],[p(x)])
prob = NeuralPDE.discretize(pde_system,discretization)
discretized_functions = NeuralPDE.discretize_inner_functions(pde_system,discretization)

pde_inner_loss_functions = discretized_functions.pde_loss_functions
bcs_inner_loss_functions = discretized_functions.bc_loss_functions

phi = discretization.phi

cb_ = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    println("additional_loss: ", norm_loss_function(phi,p,nothing))
    return false
end

res = GalacticOptim.solve(prob,LBFGS(),maxiters=400)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,BFGS(),maxiters=2000)

C = 142.88418699042
analytic_sol_func(x) = C*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.u)) for x in xs]

@test u_predict ≈ u_real rtol = 1e-3

# plot(xs ,u_real, label = "analytic")
# plot!(xs ,u_predict, label = "predict")

## Example 8, Lorenz System (Parameter Estimation)
println("Example 8, Lorenz System")

Random.seed!(1234)
@parameters t ,σ_ ,β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β*z(t)]


bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0,1.0)]
dt = 0.05

input_ = length(domains)
n = 12
chain = [FastChain(FastDense(input_,n,Flux.tanh),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:3]
#Generate Data
function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,1.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob, Tsit5(), dt=0.1)
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]

function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us,ts_]
end

data = getData(sol)

#Additional Loss Function
initθs = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
acum =  [0;accumulate(+, length.(initθs))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
(u_ , t_) = data
len = length(data[2])

function additional_loss(phi, θ , p)
    return sum(sum(abs2, phi[i](t_ , θ[sep[i]]) .- u_[[i], :])/len for i in 1:1:3)
end

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dt);
                                             init_params =initθs,
                                             param_estim=true,
                                             additional_loss=additional_loss)
testθ =reduce(vcat,initθs)
additional_loss(discretization.phi, testθ, nothing)

@named pde_system = PDESystem(eqs,bcs,domains,
                      [t],[x(t), y(t), z(t)],[σ_, ρ, β],
                      defaults=Dict([p => 1.0 for p in [σ_, ρ, β]]))
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
discretized_functions = NeuralPDE.discretize_inner_functions(pde_system,discretization)
discretized_functions.full_loss_function([testθ;ones(3)], Float64[])

res = GalacticOptim.solve(prob, Optim.BFGS(); maxiters=6000)
p_ = res.minimizer[end-2:end]
@test sum(abs2, p_[1] - 10.00) < 0.1
@test sum(abs2, p_[2] - 28.00) < 0.1
@test sum(abs2, p_[3] - (8/3)) < 0.1
#Plotting the system
# initθ = discretization.init_params
# acum =  [0;accumulate(+, length.(initθ))]
# sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
# minimizers = [res.minimizer[s] for s in sep]
# ts = [infimum(d.domain):dt/10:supremum(d.domain) for d in domains][1]
# u_predict  = [[discretization.phi[i]([t],minimizers[i])[1] for t in ts] for i in 1:3]
# plot(sol)
# plot!(ts, u_predict, label = ["x(t)" "y(t)" "z(t)"])

## Approximation of function 1D
println("Approximation of function 1D")

@parameters x
@variables u(..)

func(x) = @. 2 + abs(x - 0.5)

eq = [u(x) ~ func(x)]
bc = [u(0)~u(0)]

x0 = 0
x_end = 2
dx= 0.001
domain = [x ∈ Interval(x0,x_end)]

xs = collect(x0:dx:x_end)
func_s = func(xs)

hidden =10
chain = FastChain(FastDense(1,hidden, Flux.tanh),
                    FastDense(hidden, hidden, Flux.tanh),
                    FastDense(hidden, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain,strategy; initial_params=initθ)
@named pde_system = PDESystem(eq,bc,domain,[x],[u(x)])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

res  = GalacticOptim.solve(prob,ADAM(0.05),maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,BFGS(initial_stepnorm = 0.01),maxiters=500)

@test discretization.phi(xs',res.u) ≈ func(xs') rtol = 0.01

# plot(xs,func(xs))
# plot!(xs, discretization.phi(xs',res.u)')

## Approximation of function 1D 2
println("Approximation of function 1D 2")

@parameters x
@variables u(..)
func(x) =  @. cos(5pi*x)*x
eq = [u(x) ~ func(x)]
bc = [u(0)~u(0)]

x0 = 0
x_end = 4
domain = [x ∈ Interval(x0,x_end)]

hidden =20
chain = FastChain(FastDense(1,hidden, Flux.sin),
                  FastDense(hidden, hidden, Flux.sin),
                  FastDense(hidden, hidden, Flux.sin),
                  FastDense(hidden, 1))
initθ = DiffEqFlux.initial_params(chain)

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain,strategy; initial_params=initθ)
@named pde_system = PDESystem(eq,bc,domain,[x],[u(x)])
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
prob = NeuralPDE.discretize(pde_system,discretization)

res  = GalacticOptim.solve(prob,ADAM(0.01),maxiters=500)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,BFGS(),maxiters=1000)

dx= 0.01
xs = collect(x0:dx:x_end)
func_s = func(xs)

@test discretization.phi(xs',res.u) ≈ func(xs') rtol = 0.01

# plot(xs,func(xs))
# plot!(xs, discretization.phi(xs',res.u)')

## Approximation of function 2D
println("Approximation of function 2D")

@parameters x,y
@variables u(..)
func(x,y) =  -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))
eq = [u(x,y) ~ func(x,y)]
bc = [u(0,0) ~ u(0,0)]

x0 = -10
x_end = 10
y0 = -10
y_end = 10
d = 0.4

domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]

hidden =25
chain = FastChain(FastDense(2,hidden, Flux.tanh),
                  FastDense(hidden, hidden, Flux.tanh),
                  FastDense(hidden, hidden, Flux.tanh),
                  FastDense(hidden, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy = NeuralPDE.GridTraining(d)
discretization = NeuralPDE.PhysicsInformedNN(chain,strategy; initial_params=initθ)
@named pde_system = PDESystem(eq,bc,domain,[x,y],[u(x, y)])
prob = NeuralPDE.discretize(pde_system,discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system,discretization)
discretized_functions = NeuralPDE.discretize_inner_functions(pde_system,discretization)
discretized_functions.full_loss_function(initθ, Float64[])

res  = GalacticOptim.solve(prob,ADAM(0.01),maxiters=500)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,BFGS(),maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,BFGS(),maxiters=500)
phi = discretization.phi

xs = collect(x0:0.1:x_end)
ys = collect(y0:0.1:y_end)
u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real rtol = 0.05

# p1 = plot(xs, ys, u_real, st=:surface,title = "analytic");
# p2 = plot(xs, ys, u_predict, st=:surface,title = "predict");
# p3 = plot(xs, ys, diff_u,st=:surface,title = "error");
# plot(p1,p2,p3)

## approximation from data
println("Approximation of function from data and additional_loss")

@parameters x
@variables u(..)
eq = [u(0) ~ u(0)]
bc = [u(0) ~ u(0)]
x0 = 0
x_end = pi
dx =pi/10
domain = [x ∈ Interval(x0,x_end)]

hidden =10
chain = FastChain(FastDense(1,hidden, Flux.tanh),
                  FastDense(hidden, hidden, Flux.sin),
                  FastDense(hidden, hidden, Flux.tanh),
                  FastDense(hidden, 1))

initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy = NeuralPDE.GridTraining(dx)
xs = collect(x0:dx:x_end)'
aproxf_(x) = @. cos(pi*x)
data =aproxf_(xs)

function additional_loss_(phi, θ , p)
    sum(abs2,phi(xs,θ) .- data)
end

discretization = NeuralPDE.PhysicsInformedNN(chain,strategy;
                                             initial_params=initθ,
                                             additional_loss=additional_loss_)

phi = discretization.phi
phi(xs, initθ)
additional_loss_(phi, initθ , nothing)

@named pde_system = PDESystem(eq,bc,domain,[x],[u(x)])
prob = NeuralPDE.discretize(pde_system,discretization)

res  = GalacticOptim.solve(prob,ADAM(0.01),maxiters=500)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,BFGS(),maxiters=500)

@test phi(xs,res.u) ≈ aproxf_(xs) rtol = 0.01

# xs_ = xs'
# plot(xs_,data')
# plot!(xs_, phi(xs,res.u)')

# func(x,y) = -20.0 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
# func(x,y) = -abs(sin(x) * cos(y) * exp(abs(1 - (sqrt(x^2 + y^2)/pi))))
