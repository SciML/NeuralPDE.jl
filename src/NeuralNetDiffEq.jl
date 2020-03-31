module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase

using Flux, Zygote, DiffEqSensitivity, ForwardDiff, Random
using DiffEqFlux, Adapt
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

struct KolmogorovPDEProblem{ Mu, Sigma, Phi, X , T , D ,P} <: DiffEqBase.DEProblem
    μ::Mu
    sigma::Sigma
    phi::Phi
    xspan::Tuple{X,X}
    tspan::Tuple{T,T}
    d::D
    p::P
    KolmogorovPDEProblem( μ, sigma, phi , xspan , tspan , d, p=nothing) = new{typeof(μ),typeof(sigma),typeof(phi),eltype(tspan),eltype(xspan),typeof(d),typeof(p)}(μ,sigma,phi,xspan,tspan,d,p)
end

Base.summary(prob::KolmogorovPDEProblem) = string(nameof(typeof(prob)))
function Base.show(io::IO, A::KolmogorovPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  print(io,"xspan: ")
  show(io,A.xspan)
  println(io , "μ")
  show(io , A.μ)
  println(io,"Sigma")
  show(io , A.sigma)
end

struct GeneranNNPDEProblem{PF,BC,IC,T,X,DT,DX,P} <:DiffEqBase.DEProblem
  pde_func::PF
  boundary_conditions::BC
  initial_conditions::IC
  tspan::Tuple{T,T}
  xspan::Tuple{X,X}
  dt::DT
  dx::DX
  p::P
  GeneranNNPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan,xspan,dt,dx,p=nothing) = new{
                                                       typeof(pde_func),
                                                       typeof(boundary_conditions),typeof(initial_conditions),
                                                       eltype(tspan), eltype(xspan),
                                                       typeof(dt),typeof(dx),
                                                       typeof(p)}(
                                                       pde_func,boundary_conditions,initial_conditions,tspan,xspan,dt,dx,p)
end
Base.summary(prob::GeneranNNPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::GeneranNNPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end
struct GeneranNNTwoDimPDEProblem{PF,BC,IC,T,X,Y,DT,DX,DY,P} <:DiffEqBase.DEProblem
  pde_func::PF
  boundary_conditions::BC
  initial_conditions::IC
  tspan::Tuple{T,T}
  xspan::Tuple{X,X}
  yspan::Tuple{Y,Y}
  dt::DT
  dx::DX
  dy::DX
  p::P
  GeneranNNTwoDimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan,xspan,yspan,dt,dx,dy,p=nothing) = new{
                                                       typeof(pde_func),
                                                       typeof(boundary_conditions),typeof(initial_conditions),
                                                       eltype(tspan), eltype(xspan),eltype(yspan),
                                                       typeof(dt),typeof(dx),typeof(dy),
                                                       typeof(p)}(
                                                       pde_func,boundary_conditions,initial_conditions,tspan,xspan,yspan,dt,dx,dy,p)
end
Base.summary(prob::GeneranNNTwoDimPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::GeneranNNTwoDimPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")
include("kolmogorov_solve.jl")
include("general_ode_solve.jl")
include("general_pde_solve.jl")
include("general_two_dim_pde_solve.jl")


export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS,
        KolmogorovPDEProblem, NNKolmogorov, NNGenODE, NNGeneralPDE, GeneranNNPDEProblem, GeneranNNTwoDimPDEProblem



end # module
