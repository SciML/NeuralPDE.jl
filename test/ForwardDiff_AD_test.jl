using Flux
println("ForwardDiff_AD_tests")
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
import ModelingToolkit: Interval, infimum, supremum

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

##  1D ode
function test_ode(strategy_)
    println("Example 1, 1D ode: strategy: $(nameof(typeof(strategy_)))")
    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)
    eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))
    bcs = [u(0.) ~ 1.0]
    domains = [θ ∈ Interval(0.0,1.0)]
    chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))
    initθ = Float64.(DiffEqFlux.initial_params(chain))

    discretization = NeuralPDE.PhysicsInformedNN(chain,
                                                 strategy_;
                                                 init_params = initθ,
                                                 phi = nothing,
                                                 derivative = nothing,
												 AD=true
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

    @test u_predict ≈ u_real atol = 10^4
    # using Plots
    # t_plot = collect(ts)
    # plot(t_plot ,u_real)
    # plot!(t_plot ,u_predict)
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
strategies = [grid_strategy,stochastic_strategy,quasirandom_strategy,quasirandom_strategy_resampling] #quadrature_strategy
#TODO # quadrature_strategy get NaN

# map(strategies) do strategy_
#     test_ode(strategy_)
# end
#
# test_ode(quadrature_strategy)

println("3rd-order ode")
@parameters x
@variables u(..)
Dxxx = Differential(x)^3
Dx = Differential(x)
eq = Dxxx(u(x)) ~ cos(pi*x)
bcs= [u(0.) ~ 0.0,
      u(1.) ~ cos(pi),
      Dx(u(1.)) ~ 1.0]
domains = [x ∈ Interval(0.0,1.0)]
chain = FastChain(FastDense(1,12,Flux.tanh),FastDense(12,12,Flux.tanh),FastDense(12,1))
strategy = NeuralPDE.GridTraining(0.1)
initθ = Float64.(DiffEqFlux.initial_params(chain))
discretization = NeuralPDE.PhysicsInformedNN(chain,strategy;init_params = initθ, AD=true)
@named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
inner_loss =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1]
inner_loss(initθ)
inner_lossbcs =prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents[1]
inner_lossbcs(initθ)
prob.f(initθ, nothing)
res = GalacticOptim.solve(prob, BFGS();cb=cb, maxiters=1000)
phi = discretization.phi

analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)
xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]
@test u_predict ≈ u_real rtol = 10^-5

# x_plot = collect(xs)
# plot(x_plot ,u_real)
# plot!(x_plot ,u_predict)

## 1D PDE
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)
bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
	   u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
domains = [x ∈ Interval(0.0,1.0),
		   y ∈ Interval(0.0,1.0)]
chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,1))
initθ = Float64.(DiffEqFlux.initial_params(chain_))
strategy_ = NeuralPDE.GridTraining(0.1)
# quasirandom_strategy = NeuralPDE.QuasiRandomTraining(50;sampling_alg = LatticeRuleSample())
# stochastic_strategy = NeuralPDE.StochasticTraining(50;)
discretization = NeuralPDE.PhysicsInformedNN(chain_,
											 strategy_;
											 init_params = initθ,
											 AD=true)

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
prob = NeuralPDE.discretize(pde_system,discretization)
prob.f(initθ, nothing)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
# res = GalacticOptim.solve(prob, ADAM(0.1);cb=cb, maxiters=500)
# prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob, BFGS();cb=cb, maxiters=1000)
phi = discretization.phi
phi(rand(2,10),initθ)

xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real rtol = 10^-2

# p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
# p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)

##Kuramoto–Sivashinsky equation
@parameters x, t
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dx2 = Differential(x)^2
Dx3 = Differential(x)^3
Dx4 = Differential(x)^4

α = 1
β = 4
γ = 1
eq = Dt(u(x,t)) + u(x,t)*Dx(u(x,t)) + α*Dx2(u(x,t)) + β*Dx3(u(x,t)) + γ*Dx4(u(x,t)) ~ 0

u_analytic(x,t;z = -x/2+t) = 11 + 15*tanh(z) -15*tanh(z)^2 - 15*tanh(z)^3
du(x,t;z = -x/2+t) = 15/2*(tanh(z) + 1)*(3*tanh(z) - 1)*sech(z)^2

bcs = [u(x,0) ~ u_analytic(x,0),
       u(-10,t) ~ u_analytic(-10,t),
       u(10,t) ~ u_analytic(10,t),
       Dx(u(-10,t)) ~ du(-10,t),
       Dx(u(10,t)) ~ du(10,t)]
domains = [x ∈ Interval(-10.0,10.0),
           t ∈ Interval(0.0,1.0)]

dx = 0.4; dt = 0.2
strategy = NeuralPDE.GridTraining([dx,dt])
chain = FastChain(FastDense(2,8,Flux.σ),FastDense(8,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
discretization = NeuralPDE.PhysicsInformedNN(chain,strategy ,init_params = initθ, AD =true)
@named pde_system = PDESystem(eq,bcs,domains,[x,t],[u(x, t)])
prob = NeuralPDE.discretize(pde_system,discretization)
prob.f(initθ, nothing)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=200)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=100)
phi = discretization.phi

xs,ts = [infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,[dx/10,dt])]
u_predict = [[first(phi([x,t],res.minimizer)) for x in xs] for t in ts]
u_real = [[u_analytic(x,t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x,t) -first(phi([x,t],res.minimizer)))  for x in xs] for t in ts]

@test u_predict ≈ u_real rtol = 10^-2

# p1 =plot(xs,u_predict,title = "predict")
# p2 =plot(xs,u_real,title = "analytic")
# p3 =plot(xs,diff_u,title = "error")
# plot(p1,p2,p3)


println(" system of pde")
@parameters x, y
@variables u1(..), u2(..)
Dx = Differential(x)
Dy = Differential(y)
eqs = [Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
      Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]
      # 3*u1(x,0) ~ 2*u2(x,0)]
bcs = [u1(x,0) ~ 2*x, u2(x,0) ~ 3*x]
domains = [x ∈ Interval(0.0,1.0), y ∈ Interval(0.0,1.0)]
chain1 = FastChain(FastDense(2,5,Flux.tanh),FastDense(5,1))
chain2 = FastChain(FastDense(2,5,Flux.tanh),FastDense(5,1))

# quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
#                                                     reltol=1e-3,abstol=1e-3,
#                                                     maxiters =50, batch=100)

strategy = NeuralPDE.GridTraining(0.1)
chain = [chain1,chain2]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain,strategy; init_params = initθ, AD=true)

@named pde_system = PDESystem(eqs,bcs,domains,[x,y],[u1(x, y),u2(x, y)])

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

flat_initθ = reduce(vcat,initθ)
prob.f(flat_initθ, nothing)
# ForwardDiff.gradient(θ -> prob.f(θ, nothing),flat_initθ)

res = GalacticOptim.solve(prob,BFGS();cb=cb, maxiters=1000)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob,BFGS();cb=cb, maxiters=1000)
phi = discretization.phi

analytic_sol_func(x,y) =[1/3*(6x - y), 1/2*(6x - y)]
xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_real  = [[analytic_sol_func(x,y)[i] for x in xs  for y in ys] for i in 1:2]

initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
u_predict  = [[phi[i]([x,y],minimizers[i])[1] for x in xs  for y in ys] for i in 1:2]

error_  = u_predict .- u_real
@test u_predict[1] ≈ u_real[1] rtol = 10^-4
@test u_predict[2] ≈ u_real[2] rtol = 10^-4

# p1 =plot(xs, ys, u_predict, st=:surface);
# p2 = plot(xs, ys, u_real, st=:surface);
# p3 = plot(xs, ys, error_[2],st=:surface);
# plot(p1,p2,p3)

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

chain = [FastChain(FastDense(3,9,Flux.tanh),FastDense(9,1)),
         FastChain(FastDense(2,9,Flux.tanh),FastDense(9,1)),
         FastChain(FastDense(1,9,Flux.tanh),FastDense(9,1)),
         FastChain(FastDense(2,9,Flux.tanh),FastDense(9,1))]

initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))


grid_strategy = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain,grid_strategy;init_params = initθ,AD=true)

@named pde_system = PDESystem(eqs,bcs,domains,[x,y,z],[u(x,y,z),v(y,x),h(z),p(x,z)])

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
sym_prob[1][end]
  #   u(x,y,z) + v(y,x)*Dz(h(z)) - p(x,z) ~ x+y+z - (x^2+y^2)*sin(z) - exp(x)*exp(z)
  # (+).((+).((*).(-1, u(cord4, var"##θ#4214", phi4)),
  # (*).(derivative(Main.NeuralPDE.var"#694#697"(), var"##θ#4213", z),
  # u(cord2, var"##θ#4212", phi2))), u(cord1, var"##θ#4211", phi1))
  # .- (+).((+).((+).((+).(x, y), z), (*).((*).(-1, (exp).(x)), (exp).(z))),
  # (*).((*).(-1, (+).((^).(x, 2), (^).(y, 2))), (sin).(z)))

flat_initθ = reduce(vcat,initθ)
prob.f(flat_initθ, nothing)
# ForwardDiff.gradient(θ -> prob.f(θ, nothing),flat_initθ)

res = GalacticOptim.solve(prob, BFGS();cb=cb,maxiters=1000)
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
    @test predict[i] ≈ real_[i] rtol = 10^-2
end
#
# x_plot = collect(xs)
# y_plot = collect(ys)
# i=1
# z=0
# u_real = collect(analytic_sol_func_[1](x,y,z) for y in ys, x in xs);
# u_predict = collect(phi[1]([x,y,z],minimizers[1])[1]  for y in ys, x in xs);
# plot(x_plot,y_plot,u_real)
# plot!(x_plot,y_plot,u_predict)
