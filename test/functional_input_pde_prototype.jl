using Flux
using DiffEqFlux
using ModelingToolkit
using Test, NeuralPDE
using GalacticOptim
using SciMLBase
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# number of eigenfunctions to include in the initial condition function basis
n = 4
@parameters t x
# this is a hacky workaround to generate a vector of named a_i parameters since NeuralPDE doesn't support SymbolicArray variables yet
a_vars = eval(Symbolics._parse_vars(:parameters,
                        Real,
                        [Symbol("a_$i") for i in 1:n],
                        ModelingToolkit.toparam,
                        ))
all_vars = vcat([t, x], a_vars)
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# heat equation pde
k = 1
eq = Dt(u(t, x, a_vars...)) ~ k*Dxx(u(t, x, a_vars...))

# these eigenfunctions assume [0, 1] x domain
function dirichlet_laplacian_eigenfunction(a_var, x_var, mode_num)
    a_var * sin(x_var * π * (mode_num + 1))
end

functional_symbolic_initial_condition = sum(dirichlet_laplacian_eigenfunction(a_vars[i], x, i - 1) for i in 1:n)

# Initial and boundary conditions
bcs = [u(t,0, a_vars...) ~ 0.,# for all t > 0
       u(t,1, a_vars...) ~ 0.,# for all t > 0
       u(0,x, a_vars...) ~ functional_symbolic_initial_condition, #for all 0 < x < 1
       ]

# Space and time domains
a_domains = [a_var ∈ Interval(-1.0, 1.0) for a_var in a_vars]
domains = vcat(
    [t ∈ Interval(0.0,1.0),
     x ∈ Interval(0.0,1.0)], 
     a_domains)


@named pde_system = PDESystem(eq,bcs,domains,vcat([t,x], a_vars),[u(t, x, a_vars...)])

hid = 32
# input dimension is 2 for (t, x) and one dimension for each a_i variable
chain_ = FastChain(FastDense(2 + n,hid,Flux.σ),FastDense(hid,hid,Flux.σ),FastDense(hid,1))
strategy_ = StochasticTraining(128, 32)

initθ = Float64.(DiffEqFlux.initial_params(chain_))
discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                strategy_;
                                                init_params = initθ)

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, ADAM(1e-3); maxiters=1000)

