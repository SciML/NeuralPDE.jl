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

map(strategies) do strategy_
    test_ode(strategy_)
end

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
chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
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
res = GalacticOptim.solve(prob, ADAM(0.1);cb=cb, maxiters=500)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob, BFGS();cb=cb, maxiters=1000)
phi = discretization.phi

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

#TODO too long
# ##Kuramoto–Sivashinsky equation
# @parameters x, t
# @variables u(..)
# Dt = Differential(t)
# Dx = Differential(x)
# Dx2 = Differential(x)^2
# Dx3 = Differential(x)^3
# Dx4 = Differential(x)^4
#
# α = 1
# β = 4
# γ = 1
# eq = Dt(u(x,t)) + u(x,t)*Dx(u(x,t)) + α*Dx2(u(x,t)) + β*Dx3(u(x,t)) + γ*Dx4(u(x,t)) ~ 0
#
# u_analytic(x,t;z = -x/2+t) = 11 + 15*tanh(z) -15*tanh(z)^2 - 15*tanh(z)^3
# du(x,t;z = -x/2+t) = 15/2*(tanh(z) + 1)*(3*tanh(z) - 1)*sech(z)^2
#
# bcs = [u(x,0) ~ u_analytic(x,0),
#        u(-10,t) ~ u_analytic(-10,t),
#        u(10,t) ~ u_analytic(10,t),
#        Dx(u(-10,t)) ~ du(-10,t),
#        Dx(u(10,t)) ~ du(10,t)]
# domains = [x ∈ Interval(-10.0,10.0),
#            t ∈ Interval(0.0,1.0)]
#
# dx = 0.4; dt = 0.2
# strategy = NeuralPDE.GridTraining([dx,dt])
# chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
# initθ = Float64.(DiffEqFlux.initial_params(chain))
# discretization = NeuralPDE.PhysicsInformedNN(chain,strategy ,init_params = initθ, AD =true)
# @named pde_system = PDESystem(eq,bcs,domains,[x,t],[u(x, t)])
# prob = NeuralPDE.discretize(pde_system,discretization)
# prob.f(initθ, nothing)
# sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
# res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=200)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=100)
# phi = discretization.phi
#
# xs,ts = [infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,[dx/10,dt])]
# u_predict = [[first(phi([x,t],res.minimizer)) for x in xs] for t in ts]
# u_real = [[u_analytic(x,t) for x in xs] for t in ts]
# diff_u = [[abs(u_analytic(x,t) -first(phi([x,t],res.minimizer)))  for x in xs] for t in ts]
#
# @test u_predict ≈ u_real rtol = 10^-2
#
# # p1 =plot(xs,u_predict,title = "predict")
# # p2 =plot(xs,u_real,title = "analytic")
# # p3 =plot(xs,diff_u,title = "error")
# # plot(p1,p2,p3)
