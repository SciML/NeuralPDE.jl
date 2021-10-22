using Flux
println("Integro Differential equation tests")
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
using OrdinaryDiffEq
import ModelingToolkit: Interval, infimum, supremum
using DomainSets

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

#Integration Tests
println("Integral Tests")
@parameters t
@variables i(..)
Di = Differential(t)
Ii = Integral(t in DomainSets.ClosedInterval(0, t))
eq = Di(i(t)) + 2*i(t) + 5*Ii(i(t)) ~ 1
bcs = [i(0.) ~ 0.0]
domains = [t ∈ Interval(0.0,2.0)]
chain = Chain(Dense(1,15,Flux.σ),Dense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
strategy_ = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy_;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             )
@named pde_system = PDESystem(eq,bcs,domains,[t],[i(t)])
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=100)
ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
phi = discretization.phi

analytic_sol_func(t) = 1/2*(exp(-t))*(sin(2*t))
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(phi([t],res.minimizer)) for t in ts]
@test Flux.mse(u_real, u_predict) < 0.001
# plot(ts,u_real)
# plot!(ts,u_predict)


## Simple Integral Test
println("Simple Integral Test")

@parameters x
@variables u(..)
Ix = Integral(x in DomainSets.ClosedInterval(0, x))
# eq = Ix(u(x)) ~ (x^3)/3
eq = Ix(u(x)*cos(x))~ (x^3)/3

bcs = [u(0.) ~ 0.0]
domains = [x ∈ Interval(0.0,1.00)]
# chain = Chain(Dense(1,15,Flux.σ),Dense(15,1))
chain = FastChain(FastDense(1,15,Flux.σ),FastDense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
strategy_ = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy_;
                                             init_params = initθ,
                                             phi = nothing,
                                             derivative = nothing,
                                             )
@named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=200)
xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
phi = discretization.phi
u_predict  = [first(phi([x],res.minimizer)) for x in xs]
u_real  = [x^2/cos(x) for x in xs]
@test Flux.mse(u_real, u_predict) < 0.001

# plot(xs,u_real)
# plot!(xs,u_predict)

#simple multidimensitonal integral test
println("simple multidimensitonal integral test")

@parameters x,y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
Ix = Integral((x,y) in DomainSets.UnitSquare())
eq = Ix(u(x,y)) ~ 1/3
bcs = [u(0., 0.) ~ 1, Dx(u(x,y)) ~ -2*x , Dy(u(x ,y)) ~ -2*y ]
domains = [x ∈ Interval(0.0,1.00), y ∈ Interval(0.0,1.00)]
chain = Chain(Dense(2,15,Flux.σ),Dense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
strategy_ = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy_;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             )
@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=100)
xs = 0.00:0.01:1.00
ys = 0.00:0.01:1.00
phi = discretization.phi

u_real = collect(1 - x^2 - y^2 for y in ys, x in xs);
u_predict = collect(Array(phi([x,y], res.minimizer))[1] for y in ys, x in xs);
@test Flux.mse(u_real, u_predict) < 0.001

# error_ = u_predict .- u_real
# p1 = plot(xs,ys,u_real,linetype=:contourf,label = "analytic")
# p2 = plot(xs,ys,u_predict,linetype=:contourf,label = "predict")
# p3 = plot(xs,ys,error_,linetype=:contourf,label = "error")
# plot(p1,p2,p3)

@parameters x,y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
Ix = Integral((x,y) in DomainSets.ProductDomain(UnitInterval(),ClosedInterval(0 ,x)))
eq = Ix(u(x,y)) ~ 5/12
bcs = [u(0., 0.) ~ 0, Dy(u(x,y)) ~ 2*y , u(x, 0) ~ x ]
domains = [x ∈ Interval(0.0,1.00), y ∈ Interval(0.0,1.00)]
chain = Chain(Dense(2,15,Flux.σ),Dense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
strategy_ = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy_;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             )
@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=100)
xs = 0.00:0.01:1.00
ys = 0.00:0.01:1.00
phi = discretization.phi

u_real = collect( x + y^2 for y in ys, x in xs);
u_predict = collect(Array(phi([x,y], res.minimizer))[1] for y in ys, x in xs);
@test Flux.mse(u_real, u_predict) < 0.01

# error_ = u_predict .- u_real
# p1 = plot(xs,ys,u_real,linetype=:contourf,label = "analytic")
# p2 = plot(xs,ys,u_predict,linetype=:contourf,label = "predict")
# p3 = plot(xs,ys,error_,linetype=:contourf,label = "error")
# plot(p1,p2,p3)
