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
import ModelingToolkit: Interval, infimum, supremum

using Random
Random.seed!(100)

cb = function (p, l)
    println("Current loss is: $l")
    return false
end

## Example 1, heterogeneous equation
function test_heterogeneous_equation(strategy_)
	println("Example 1, 2D heterogeneous equation, strategy: $(nameof(typeof(strategy_)))")
	@parameters x, y
	@variables u1(..), u2(..)
	Dx = Differential(x)
	Dy = Differential(y)

	# System of pde
	eqs = u1(x, y) + Dx(u2(x)) ~ 0

	# Initial and boundary conditions
	bcs = [u1(x, 0) ~ cos(x), u2(x) ~ 0]

	# Space and time domains
	domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]


	# Neural network
	chain1 = FastChain(FastDense(1, 15, Flux.tanh), FastDense(15, 1))
	chain2 = FastChain(FastDense(2, 15, Flux.tanh), FastDense(15, 1))

	chains = [chain1,chain2]
	initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

	discretization = NeuralPDE.PhysicsInformedNN(chains, strategy_; init_params=initθ)

	pde_system = PDESystem(eqs, bcs, domains, [x,y], [u1(x, y),u2(x)])
	sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)


	res = GalacticOptim.solve(prob, ADAM(0.1); cb=cb, maxiters=1000)
	prob = remake(prob, u0=res.minimizer)
	res = GalacticOptim.solve(prob, ADAM(0.01); cb=cb, maxiters=1000)
	prob = remake(prob, u0=res.minimizer)
	res = GalacticOptim.solve(prob, ADAM(0.001); cb=cb, maxiters=1000)
	phi = discretization.phi

	# TODO change analytic solution here of course
	analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
	ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
	u_real  = [analytic_sol_func(t) for t in ts]
	u_predict  = [first(phi(t, res.minimizer)) for t in ts]

	@test u_predict ≈ u_real atol = 0.1
end

## Example 2, heterogeneous system
function test_heterogeneous_system()
	println("Example 2, system of heterogeneous pdes: strategy: $(nameof(typeof(strategy_)))")
	println("Example 4, system of pde")
	@parameters x, y
	@variables u1(..), u2(..)
	Dx = Differential(x)
	Dy = Differential(y)

	# System of pde
	eqs = [Dx(u1(x, y)) + 4 * Dy(u2(x)) ~ 0,
		Dx(u2(x)) + 9 * Dy(u1(x, y)) ~ 0]

	# Initial and boundary conditions
	bcs = [u1(x, 0) ~ 2 * x, u2(x, 0) ~ 3 * x]

	# Space and time domains
	domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

	# Neural network
	chain1 = FastChain(FastDense(2, 15, Flux.σ), FastDense(15, 1))
	chain2 = FastChain(FastDense(2, 15, Flux.σ), FastDense(15, 1))

	strategy = NeuralPDE.QuadratureTraining()
	chain = [chain1,chain2]
	initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
	discretization = NeuralPDE.PhysicsInformedNN(chain, strategy; init_params=initθ)
	pde_system = PDESystem(eqs, bcs, domains, [x,y], [u1,u2])
	prob = NeuralPDE.discretize(pde_system, discretization)
	sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

	res = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=1000)
	phi = discretization.phi

	analytic_sol_func(x, y) = [1 / 3 * (6x - y), 1 / 2 * (6x - y)]
	xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
	u_real  = [[analytic_sol_func(x, y)[i] for x in xs  for y in ys] for i in 1:2]

	initθ = discretization.init_params
	acum =  [0;accumulate(+, length.(initθ))]
	sep = [acum[i] + 1:acum[i + 1] for i in 1:length(acum) - 1]
	minimizers = [res.minimizer[s] for s in sep]
	u_predict  = [[phi[i]([x,y], minimizers[i])[1] for x in xs  for y in ys] for i in 1:2]

	@test u_predict[1] ≈ u_real[1] atol = 0.1
	@test u_predict[2] ≈ u_real[2] atol = 0.1
end
    
    
grid_strategy = NeuralPDE.GridTraining(0.1)
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                        reltol=1e-3,abstol=1e-3,
                                                        maxiters=50, batch=100)
stochastic_strategy = NeuralPDE.StochasticTraining(400; bcs_points=50) # points
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(400; # points
                                                         sampling_alg=LatinHypercubeSample(),
                                                         resampling=false,
                                                         minibatch=100
                                                        )
quasirandom_strategy_resampling = NeuralPDE.QuasiRandomTraining(400; # points
                                                         bcs_points=50,
                                                         sampling_alg=LatinHypercubeSample(),
                                                         resampling=true,
                                                         minibatch=0)
    
strategies = [grid_strategy,stochastic_strategy, quadrature_strategy,quasirandom_strategy,quasirandom_strategy_resampling]
    
for strategy_ in strategies
	test_heterogeneous_equation(strategy_)
end
    
test_heterogeneous_system()
