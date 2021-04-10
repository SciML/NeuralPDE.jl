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
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))
initθ = DiffEqFlux.initial_params(chain)

strategy = NeuralPDE.StochasticTraining(128)
adaloss = NeuralPDE.LossGradientsAdaptiveLoss(20)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             adaptive_loss=adaloss,
                                             )

pde_system = PDESystem(eq,bcs,domains,[θ],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(1e-3); cb = cb, maxiters=5000)
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