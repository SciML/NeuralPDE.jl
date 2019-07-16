module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase
using Flux

abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
struct nnode{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
end
nnode(chain;opt=Flux.ADAM(0.1)) = nnode(chain,opt)
export nnode


include("solve.jl")
include("PdeSolve.jl")


end # module
