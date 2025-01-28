"""
    BayesianPINN(args...; dataset = nothing, kwargs...)

A `discretize` algorithm for the ModelingToolkit PDESystem interface, which transforms a
`PDESystem` into a likelihood function used for HMC based Posterior Sampling Algorithms
[AdvancedHMC.jl](https://turinglang.org/AdvancedHMC.jl/stable/) which is later optimized
upon to give the Solution Distribution of the PDE, using the Physics-Informed Neural
Networks (PINN) methodology.

All positional arguments and keyword arguments are passed to `PhysicsInformedNN` except
the ones mentioned below.

## Keyword Arguments

* `dataset`: A vector of matrix, each matrix for ith dependant variable and first col in
  matrix is for dependant variables, remaining columns for independent variables. Needed for
  inverse problem solving.
"""
@concrete struct BayesianPINN <: AbstractPINN
    pinn <: PhysicsInformedNN
    dataset
end

function Base.getproperty(pinn::BayesianPINN, name::Symbol)
    name === :dataset && return getfield(pinn, :dataset)
    name === :pinn && return getfield(pinn, :pinn)
    return getproperty(pinn.pinn, name)
end

function BayesianPINN(args...; dataset = nothing, kwargs...)
    dataset === nothing && (dataset = (nothing, nothing))
    return BayesianPINN(PhysicsInformedNN(args...; kwargs...), dataset)
end