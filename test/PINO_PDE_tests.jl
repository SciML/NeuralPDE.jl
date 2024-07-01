using Test
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Lux
using Statistics, Random
using ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using NeuralPDE


##example ODE
@parameters t
@variables u(..)
# @parameters p #[bounds = (0.1f0, pi)]
Dt = Differential(t)
eq = [Dt(u(t)) ~ cos(t)]
bc = [u(0) ~ 1.0f0]

dom = [x ∈ Interval(0.0, 1.0)]
# neural_operator = SomeNeuralOperator(some_args)
neural_operator = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh),
    Lux.Dense(10, 10, Lux.tanh),
    Lux.Dense(10, 1))

# pino = PhysicsInformedNO(neural_operator, sometrainig)
pino = NeuralPDE.PhysicsInformedNN(neural_operator, NeuralPDE.GridTraining(0.1))

@named pde_system = PDESystem(eq, bc, dom, [t], [u(t)])  #[p]; defaults = Dict([p => 1.0 for p in [p]]))

# hasbounds(pde_system.ps[1])
# getbounds(pde_system.ps[1])

prob = discretize(pde_system, pino)
sym_prob = symbolic_discretize(pde_system, pino)

res = Optimization.solve(prob, ADAM(0.1); maxiters = 4000)
phi = discretization.phi


@parameters x
@parameters p [bounds = (0.1f0, pi)]
@variables u(..)
Dx = Differential(x)

eq = Dx(u(x)) ~ cos(p * x)
bcs = [u(0.0) ~ 0.0]

domains = [x ∈ Interval(0.0, 1.0)]

chain = [Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh), Lux.Dense(12, 1)) ]

strategy = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)],[p]; defaults = Dict([p => 1.0 for p in [p]]))
hasbounds(pde_system.ps[1])
getbounds(pde_system.ps[1])

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)
