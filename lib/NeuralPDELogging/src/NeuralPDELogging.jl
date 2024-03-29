module NeuralPDELogging

using NeuralPDE
using TensorBoardLogger

"""This function overrides the empty function in NeuralPDE in order to use TensorBoardLogger in that package
This is light type piracy but it should be alright since this is a subpackage of NeuralPDE"""
function NeuralPDE.logvector(logger::TBLogger, vector::AbstractVector{R},
        name::AbstractString, step::Integer) where {R <: Real}
    for j in 1:length(vector)
        log_value(logger, "$(name)/$(j)", vector[j], step = step)
    end
    nothing
end

"""This function overrides the empty function in NeuralPDE in order to use TensorBoardLogger in that package.  
This is light type piracy but it should be alright since this is a subpackage of NeuralPDE"""
function NeuralPDE.logscalar(logger::TBLogger, scalar::R, name::AbstractString,
        step::Integer) where {R <: Real}
    log_value(logger, "$(name)", scalar, step = step)
    nothing
end

end
