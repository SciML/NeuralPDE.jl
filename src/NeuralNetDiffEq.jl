module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase
using Flux, Zygote, DiffEqSensitivity, ForwardDiff
using DiffEqFlux
import Tracker, Optim

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

struct GeneranNNPDEProblem{PF,BC,T,X,DT,DX,P} <:DiffEqBase.DEProblem
  pde_func::PF
  boundary_conditions::BC
  tspan::Tuple{T,T}
  xspan::Tuple{X,X}
  dt::DT
  dx::DX
  p::P
  GeneranNNPDEProblem(pde_func,boundary_conditions,tspan,xspan,dt,dx,p=nothing) = new{
                                                       typeof(pde_func),typeof(boundary_conditions),
                                                       eltype(tspan), eltype(xspan),
                                                       typeof(dt),typeof(dx),
                                                       typeof(p)}(
                                                       pde_func,boundary_conditions,tspan,xspan,dt,dx,p)
end
Base.summary(prob::GeneranNNPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::GeneranNNPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")
include("general_ode_solve.jl")
include("general_pde_solve.jl")

export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS,
       NNGenODE, NNGeneralPDE, GeneranNNPDEProblem

end # module
