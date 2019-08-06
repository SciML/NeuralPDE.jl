module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase
using Flux

abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct TerminalPDEProblem{G,F,Mu,Sigma,X,T,P} <: DiffEqBase.DEProblem
    g::G
    f::F
    μ::Mu
    σ::Sigma
    X0::X
    tspan::Tuple{T,T}
    p::P
    TerminalPDEProblem(g,f,μ,σ,X0,tspan,p=nothing) = new{typeof(g),typeof(f),
                                                         typeof(μ),typeof(σ),
                                                         typeof(X0),eltype(tspan),
                                                         typeof(p)}(
                                                         g,f,μ,σ,X0,tspan,p)
end

include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")


export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS

end # module
