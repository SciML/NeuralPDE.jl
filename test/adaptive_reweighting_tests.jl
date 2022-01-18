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
#using OrdinaryDiffEq
using Plots
import ModelingToolkit: Interval, infimum, supremum
using DomainSets

end
begin
using Random

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 2, 2D Poisson equation
end
function test_2d_poisson_equation(adaptive_loss, seed=100)
    Random.seed!(seed)
    hid = 40
    chain_ = FastChain(FastDense(2,hid,Flux.σ),FastDense(hid,hid,Flux.σ),FastDense(hid,1))
    alg = CubatureJLp() #CubatureJLh(),
    #strategy_ =  NeuralPDE.QuadratureTraining(quadrature_alg = alg,reltol=1e-4,abstol=1e-3,maxiters=200, batch=10)
    strategy_ =  NeuralPDE.StochasticTraining(256)
#main()
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
                                                 init_params = initθ,
                                                 adaptive_loss = adaptive_loss)

    @named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
    prob = NeuralPDE.discretize(pde_system,discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
    res = GalacticOptim.solve(prob, ADAM(0.03); maxiters=4000)
    phi = discretization.phi

    xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

    u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
    u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    diff_u = abs.(u_predict .- u_real)

    p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    plot(p1,p2,p3)
end

begin 

nonadaptive_loss = NeuralPDE.NonAdaptiveLossWeights{Float64}(pde_loss_weights=1, bc_loss_weights=1)
gradnormadaptive_loss = NeuralPDE.GradientNormAdaptiveLoss{Float64}(100, pde_loss_weights=1, bc_loss_weights=1)
adaptive_loss = NeuralPDE.MiniMaxAdaptiveLoss{Float64}(100; pde_loss_weights=1, bc_loss_weights=1)
adaptive_losses = [nonadaptive_loss, gradnormadaptive_loss,adaptive_loss]
#adaptive_losses = [adaptive_loss]

plots = map(test_2d_poisson_equation, adaptive_losses)

end
plots[1]
plots[2]
plots[3]


end