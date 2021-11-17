using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
import ModelingToolkit: Interval, infimum, supremum
using DomainSets

@parameters v x t
@variables f(..)
Iv = Integral(t in DomainSets.ClosedInterval(-Inf ,Inf))
eqs_ = [Iv(f(t, x, v)) ~ 0]

# transform
@parameters τ
function v_if(t)
    return t ./ (1 .- t.^2)
end
eqs = Symbolics.substitute(eqs_,  Dict([t => v_if(τ)]))

bcs = [f(0,x,v) ~ 2]

domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0), 
           v ∈ Interval(0.0, 1.0)]

# Neural Network
chain = [FastChain(FastDense(3, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1))]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params= initθ)
@named pde_system = PDESystem(eqs_, bcs, domains, [t,x,v], [f(t,x,v)])
prob = SciMLBase.symbolic_discretize(pde_system, discretization)
prob = SciMLBase.discretize(pde_system, discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=100)


#### messing around with get_bounds
fill((/)(τ, (+)(1, (*)(-1, (^)(τ, 2)))),2)

dict_lower_bound = Dict(:v => 0.0, :t => 0.0, :x => 0.0)
pde_args = Vector{Any}[[:((/)(τ, (+)(1, (*)(-1, (^)(τ, 2))))), :x, :v]]
pd = Any[:((/)(τ, (+)(1, (*)(-1, (^)(τ, 2))))), :x, :v]

adapt(to, x) = adapt_structure(to, x)

adapt_structure(to, x) = adapt_storage(to, x)
adapt_storage(to, x) = x
eltypeθ = Float64

pde_lower_bounds= map(pde_args) do pd
    span = map(p -> get(dict_lower_bound, p, p isa Expr ? adapt(eltype, 0) : p), pd)
    @show span # span = Any[:(τ / (1 + -1 * τ ^ 2)), 0.0, 0.0]
    map(s -> adapt(eltypeθ,eval(s)) + cbrt(eps(eltypeθ)), span)
end

pde_args2 = Vector{Any}[[:t, :x, :v]]
pd2 = Any[:t, :x, :v]
adapt(eltypeθ,:x)


pde_lower_bounds2= map(pde_args2) do pd
    span = map(p -> get(dict_lower_bound, p, 0.0), pd)
    @show span # span = [0.0, 0.0, 0.0]
    map(s -> adapt(eltypeθ,eval(s)) + cbrt(eps(eltypeθ)), span)
end

# manipulate args
@parameters t
args = :(i((+)(2, (*)(-1, (^)(t, 2))))).args
eq = Symbolics.substitute(args,  t => v_if(τ))

eval(eq[2])


# TODO
# [x] figure out right symbolic representation 
# [] fix  no method matching length(::Expr) bug
# [] incorporate into codeflow


# 
# lb, ub = get_limits(Iv.domain.domain)
# if lb || ub == Inf
    # change boundaries
    # find the integrating variable
    # replace vars in integral with to_finite(τ)
# end