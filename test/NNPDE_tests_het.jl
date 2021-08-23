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
	println("Example 1, Simple Heterogeneous input PDE, strategy: $strategy_")
	@parameters x y
	@variables p(..) q(..) r(..) s(..)
	Dx = Differential(x)
	Dy = Differential(y)

	# 2D PDE
	eq  = p(x) + q(y) + r(x, y) + s(y, x) ~ 0
	# eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

	# Initial and boundary conditions
	bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
			r(x, -1) ~ 0.f0, r(1, y) ~ 0.0f0, 
			s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
	# bcs = [s(y, 1) ~ 0.0f0]
	# Space and time domains
	domains = [x ∈ IntervalDomain(0.0, 1.0),
				y ∈ IntervalDomain(-1.0, 0.0)]

	# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
	numhid = 3
	fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
				[FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
	discretization = NeuralPDE.PhysicsInformedNN(fastchains,
													strategy_)

	pde_system = PDESystem(eq, bcs, domains, [x,y], [p(x), q(y), r(x, y), s(y, x)])
	prob = SciMLBase.discretize(pde_system, discretization)
	res = GalacticOptim.solve(prob, BFGS(); maxiters=2)
end

## Example 2, heterogeneous system
function test_heterogeneous_system(strategy_)
	println("Example 2, Heterogeneous input PDE with derivatives, strategy: $strategy_")
	@parameters x y
	@variables p(..) q(..) r(..) s(..)
	Dx = Differential(x)
	Dy = Differential(y)

	# 2D PDE
	eq = p(x) + Dx(q(y)) ~ 0

	# Initial and boundary conditions
	bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
			r(x, -1) ~ 0.f0, r(1, y) ~ 0.0f0, 
			s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
	# bcs = [s(y, 1) ~ 0.0f0]
	# Space and time domains
	domains = [x ∈ IntervalDomain(0.0, 1.0),
				y ∈ IntervalDomain(-1.0, 0.0)]

	# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
	numhid = 3
	fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
				[FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
	discretization = NeuralPDE.PhysicsInformedNN(fastchains,
													strategy_)

	pde_system = PDESystem(eq, bcs, domains, [x,y], [p(x), q(y), r(x, y), s(y, x)])
	prob = SciMLBase.discretize(pde_system, discretization)
	res = GalacticOptim.solve(prob, BFGS(); maxiters=2)
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
