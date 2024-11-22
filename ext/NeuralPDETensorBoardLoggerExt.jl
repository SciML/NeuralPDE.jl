module NeuralPDETensorBoardLoggerExt

using NeuralPDE: NeuralPDE
using TensorBoardLogger: TBLogger, log_value

function NeuralPDE.logvector(logger::TBLogger, vector::AbstractVector{<:Real},
        name::AbstractString, step::Integer)
    foreach(enumerate(vector)) do (j, v)
        log_value(logger, "$(name)/$(j)", v; step)
    end
end

function NeuralPDE.logscalar(logger::TBLogger, scalar::Real, name::AbstractString,
        step::Integer)
    log_value(logger, "$(name)", scalar; step)
    return nothing
end

end
