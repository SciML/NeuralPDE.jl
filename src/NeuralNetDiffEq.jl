module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase

using Flux, Zygote, DiffEqSensitivity, ForwardDiff, Random, Distributions
using DiffEqFlux, Adapt
using ModelingToolkit
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

struct KolmogorovPDEProblem{ F, G, Phi, X , T , D ,P,U0, ND} <: DiffEqBase.DEProblem
    f::F
    g::G
    phi::Phi
    xspan::Tuple{X,X}
    tspan::Tuple{T,T}
    d::D
    p::P
    u0::U0
    noise_rate_prototype::ND
    KolmogorovPDEProblem( f, g, phi , xspan , tspan , d, p=nothing, u0=0 , noise_rate_prototype= nothing) = new{typeof(f),typeof(g),typeof(phi),eltype(tspan),eltype(xspan),typeof(d),typeof(p),typeof(u0),typeof(noise_rate_prototype)}(f,g,phi,xspan,tspan,d,p,u0,noise_rate_prototype)
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

struct NNPDEProblem{PDESystem,MOLFiniteDifference,P} <:DiffEqBase.DEProblem
  pde_system::PDESystem
  discretization ::MOLFiniteDifference
  p::P
  NNPDEProblem(pde_system,discretization,p=nothing) = new{
                                                       typeof(pde_system),
                                                       typeof(discretization),
                                                       typeof(p)
                                                       }(pde_system,discretization,p)
end
Base.summary(prob::NNPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::NNPDEProblem)
  println(io,summary(A))
  print(io,"pde_system: ")
  show(io,A.pde_system)
  print(io,"discretization: ")
  show(io,A.discretization)
end

include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")
include("kolmogorov_solve.jl")
include("stopping_solve.jl")
include("pinns_pde_solve.jl")



export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS,
       KolmogorovPDEProblem, NNKolmogorov, NNStopping,
       NNPDE, NNPDEProblem, Spaces, Discretization

end # module
