module NeuralNetDiffEq

using Reexport
@reexport using DiffEqBase
using Knet, ForwardDiff

abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
struct nnode <: NeuralNetDiffEqAlgorithm
    hl_width::Int
end
nnode(;hl_width=10) = nnode(hl_width)
export nnode

include("solve.jl")
include("training_utils.jl")

end # module
