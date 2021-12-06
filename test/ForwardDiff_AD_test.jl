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

## Example 1, 1D ode
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
