module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase
using Flux, Zygote, DiffEqSensitivity, Distributions, Random
import Tracker

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

Base.summary(prob::TerminalPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::TerminalPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

struct KolmogorovPDEProblem{ Phi, X , T , D ,P} <: DiffEqBase.DEProblem
    phi::Phi
    xspan::Tuple{X,X}
    tspan::Tuple{T,T}
    d::D
    p::P
    KolmogorovPDEProblem( phi , xspan , tspan , d, p=nothing) = new{typeof(phi),eltype(tspan),eltype(xspan),typeof(d),typeof(p)}(phi,xspan,tspan,d,p)
end
 
Base.summary(prob::KolmogorovPDEProblem) = string(nameof(typeof(prob)))
function Base.show(io::IO, A::KolmogorovPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  print(io,"xspan: ")
  show(io,A.xspan)
  show(io , A.u0)
end
 
include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")
include("kolmogorov_solve.jl")


export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS, KolmogorovPDEProblem, NNKolmogorov

end # module
