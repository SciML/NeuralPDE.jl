module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase
using Flux

abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

include("ode_solve.jl")
include("pde_solve.jl")

export NNODE, TerminalPDEProblem, NNPDEHan

end # module
