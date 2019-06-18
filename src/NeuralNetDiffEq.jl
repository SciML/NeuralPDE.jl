module NeuralNetDiffEq

using Reexport
@reexport using DiffEqBase
using Flux

abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
struct nnode{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
end
nnode(chain,opt=Adam(0.01)) = nnode(chain,opt)
export nnode

include("solve.jl")
include("training_utils.jl")

end # module
