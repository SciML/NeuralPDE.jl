begin
push!(LOAD_PATH, "/home/zobot/.julia/dev/NeuralPDE.jl/src")
using Revise
using Flux
println("NNPDE_tests_heterogeneous")
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

using Random
end

begin
Random.seed!(100)
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 1, 1D ode
function test_heterogeneous_input(strategy_)
end

dx = 0.1
grid_strategy = NeuralPDE.GridTraining(dx)
stochastic_strategy = NeuralPDE.StochasticTraining(6) #points
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =50, batch=100)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100; #points
                                                     sampling_alg = UniformSample(),
                                                     minibatch = 100)

strategies = [grid_strategy, stochastic_strategy, quadrature_strategy,quasirandom_strategy]
strategies = [stochastic_strategy, quadrature_strategy]
#for strategy_ in strategies
    #test_heterogeneous_input(strategy_)
#end
strategy_ = strategies[1]

#=
println("Example 10, Simple Heterogeneous input PDE comparison, strategy: $strategy_")
@parameters x y
@variables r(..)
Dx = Differential(x)
Dy = Differential(y)

# 2D PDE
eq  = Dx(r(x,y)) + r(x, y) ~ 0

# Initial and boundary conditions
bcs = [
        r(x,-1) ~ 0.f0, r(1, y) ~ 0.0f0, 
        ]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
            y ∈ IntervalDomain(-1.0,0.0)]

# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
numhid = 3
fastchains = FastChain(FastDense(2,numhid,Flux.σ),FastDense(numhid,numhid,Flux.σ),FastDense(numhid,1))
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                strategy_)

pde_system = PDESystem(eq,bcs,domains,[x,y],[r(x,y)])
end
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
@run sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
prob = NeuralPDE.discretize(pde_system,discretization)
initθ = discretization.init_params
initθvec = vcat(initθ...)
prob.f(initθvec, [])
@run prob.f(initθvec, [])
=#

println("Example 10, Simple Heterogeneous input PDE, strategy: $strategy_")
@parameters x y
@variables p(..) q(..) r(..) s(..)
Dx = Differential(x)
Dy = Differential(y)

# 2D PDE
eq  = p(x) + q(y) + r(x, y) + s(y, x) ~ 0
#eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

# Initial and boundary conditions
bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
        r(x,-1) ~ 0.f0, r(1, y) ~ 0.0f0, 
        s(y,1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
#bcs = [s(y, 1) ~ 0.0f0]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
            y ∈ IntervalDomain(-1.0,0.0)]

# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
numhid = 3
fastchains = [[FastChain(FastDense(1,numhid,Flux.σ),FastDense(numhid,numhid,Flux.σ),FastDense(numhid,1)) for i in 1:2];
              [FastChain(FastDense(2,numhid,Flux.σ),FastDense(numhid,numhid,Flux.σ),FastDense(numhid,1)) for i in 1:2]]
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                strategy_)

pde_system = PDESystem(eq,bcs,domains,[x,y],[p(x), q(y), r(x,y), s(y,x)])
end




sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
@run sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
prob = NeuralPDE.discretize(pde_system,discretization)
initθ = discretization.init_params
initθvec = vcat(initθ...)
prob.f(initθvec, [])
@run prob.f(initθvec, [])
#res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=3)
phi = discretization.phi
eqs = pde_system.eqs
bcs = pde_system.bcs

domains = pde_system.domain
eq_params = pde_system.ps
defaults = pde_system.defaults
default_p = eq_params == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in eq_params]

param_estim = discretization.param_estim
additional_loss = discretization.additional_loss

# dimensionality of equation
dim = length(domains)
depvars,indvars,dict_indvars,dict_depvars,dict_depvar_input = NeuralPDE.get_vars(pde_system.indvars,pde_system.depvars)

chain = discretization.chain
initθ = discretization.init_params
flat_initθ = if (typeof(chain) <: AbstractVector) vcat(initθ...) else  initθ end
flat_initθ = if param_estim == false flat_initθ else vcat(flat_initθ, adapt(DiffEqBase.parameterless_type(flat_initθ),default_p)) end
phi = discretization.phi
derivative = discretization.derivative
strategy = discretization.strategy
if !(eqs isa Array)
    eqs = [eqs]
end
pde_indvars = if strategy isa QuadratureTraining
        NeuralPDE.get_argument(eqs,dict_indvars,dict_depvars)
else
        NeuralPDE.get_variables(eqs,dict_indvars,dict_depvars)
end
_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,
                                            dict_indvars,dict_depvars,dict_depvar_input,
                                            phi, derivative,chain, initθ,strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                            bc_indvars = pde_indvar) for (eq, pde_indvar) in zip(eqs,pde_indvars)]
bc_indvars = if strategy isa QuadratureTraining
        get_argument(bcs,dict_indvars,dict_depvars)
else
        get_variables(bcs,dict_indvars,dict_depvars)
end

_bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                dict_indvars,dict_depvars,dict_depvar_input,
                                                phi, derivative,chain, initθ, strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p;
                                                bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]