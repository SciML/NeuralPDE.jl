begin
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
using OrdinaryDiffEq
using Plots

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
end

begin
println("Example 1, 1D ode")
@parameters θ
@variables u(..)
Dθ = Differential(θ)

eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))

# Initial and boundary conditions
bcs = [u(0.) ~ 1.0]

# Space and time domains
domains = [θ ∈ IntervalDomain(0.0,1.0)]
# Discretization
dt = 0.1
# Neural network
#chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))
nonlin = Flux.tanh
numh = 36
num_hid = 1
middle_layers = [FastDense(numh, numh, nonlin) for i in 1:num_hid]
chain = FastChain(FastDense(1,numh,nonlin),middle_layers..., FastDense(numh,1))
initθ = DiffEqFlux.initial_params(chain)

strategy = NeuralPDE.StochasticTraining(128)
#strategy = NeuralPDE.QuadratureTraining()
#adaloss = NeuralPDE.LossGradientsAdaptiveLoss(20; α=0.95f0)
adaloss = NeuralPDE.NonAdaptiveLossWeights()
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             #adaptive_loss=adaloss,
                                             )

pde_system = PDESystem(eq,bcs,domains,[θ],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

opt = Flux.Optimiser(ExpDecay(1), ADAM(1e-1))
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=5000)
#=
prob2 = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob2, ADAM(0.001); cb = cb, maxiters=10)
=#
phi = discretization.phi

analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
ts = [domain.domain.lower:dt/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]

t_plot = collect(ts)
plot(t_plot ,u_real)
plot!(t_plot ,u_predict)
end



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

strategy_ = StochasticTraining(128)
chain_ = FastChain(FastDense(2,24,Flux.σ),FastDense(24,24,Flux.σ),FastDense(24,1))
adalosspoisson = NeuralPDE.LossGradientsAdaptiveLoss(20; α=0.95f0)
discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                strategy_,
                                                adaptive_loss=adalosspoisson)

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, ADAM(0.001); cb = cb, maxiters=5000)
phi = discretization.phi

dx = 0.1
xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

#@test u_predict ≈ u_real atol = 3.0

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)