"""
    LogOptions(log_frequency)
    LogOptions(; log_frequency = 50)

Options for logging during optimization.
"""
struct LogOptions
    log_frequency::Int
    # TODO: add in an option for saving plots in the log. this is currently not done because the type of plot is dependent on the PDESystem
    #       possible solution: pass in a plot function?
    #       this is somewhat important because we want to support plotting adaptive weights that depend on pde independent variables
    #       and not just one weight for each loss function, i.e. pde_loss_weights(i, t, x) and since this would be function-internal,
    #       we'd want the plot & log to happen internally as well
    #       plots of the learned function can happen in the outer callback, but we might want to offer that here too
end

LogOptions(; log_frequency = 50) = LogOptions(log_frequency)

logvector(logger, v::AbstractVector{<:Real}, name::AbstractString, step::Integer) = nothing
logscalar(logger, s::Real, name::AbstractString, step::Integer) = nothing

"""
An encoding of the test function phi that is used for calculating the PDE
value at domain points x

Fields:

- `f`: A representation of the chain function.
- `st`: The state of the Lux.AbstractLuxLayer. It should be updated on each call.
"""
@concrete struct Phi
    smodel <: StatefulLuxLayer
end

function Phi(layer::AbstractLuxLayer; init_states = nothing)
    init_states === nothing && (init_states = initialstates(Random.default_rng(), layer))
    return Phi(StatefulLuxLayer{true}(layer, nothing, init_states))
end

(f::Phi)(x::Number, θ) = only(cdev(f([x], θ)))

(f::Phi)(x::AbstractArray, θ) = f.smodel(safe_get_device(θ)(x), θ)

"""
    PhysicsInformedNN(chain, strategy; init_params = nothing, init_states = nothing,
                      phi = nothing, param_estim = false, additional_loss = nothing,
                      adaptive_loss = nothing, logger = nothing, log_options = LogOptions(),
                      iteration = nothing, kwargs...)

A `discretize` algorithm for the ModelingToolkit PDESystem interface, which transforms a
`PDESystem` into an `OptimizationProblem` using the Physics-Informed Neural Networks (PINN)
methodology.

## Positional Arguments

* `chain`: a vector of Lux/Flux chains with a d-dimensional input and a 1-dimensional output
           corresponding to each of the dependent variables. Note that this specification
           respects the order of the dependent variables as specified in the PDESystem.
           Flux chains will be converted to Lux internally using
           `adapt(FromFluxAdaptor(), chain)`.
* `strategy`: determines which training strategy will be used. See the Training Strategy
              documentation for more details.

## Keyword Arguments

* `init_params`: the initial parameters of the neural networks. If `init_params` is not
  given, then the neural network default parameters are used. Note that for Lux, the default
  will convert to Float64.
* `init_states`: the initial states of the neural networks. If `init_states` is not
  given, then the neural network default states are used. Note that for Lux, the default
  will convert to Float64.
* `phi`: a trial solution, specified as `phi(x,p)` where `x` is the coordinates vector for
  the dependent variable and `p` are the weights of the phi function (generally the weights
  of the neural network defining `phi`). By default, this is generated from the `chain`.
  This should only be used to more directly impose functional information in the training
  problem, for example imposing the boundary condition by the test function formulation.
* `adaptive_loss`: the choice for the adaptive loss function. See the
  [adaptive loss page](@ref adaptive_loss) for more details. Defaults to no adaptivity.
* `additional_loss`: a function `additional_loss(phi, θ, p_)` where `phi` are the neural
  network trial solutions, `θ` are the weights of the neural network(s), and `p_` are the
  hyperparameters of the `OptimizationProblem`. If `param_estim = true`, then `θ`
  additionally contains the parameters of the differential equation appended to the end of
  the vector.
* `param_estim`: whether the parameters of the differential equation should be included in
  the values sent to the `additional_loss` function. Defaults to `false`.
* `logger`: a logger object (e.g. from TensorBoardLogger.jl) for recording training
  metrics. Defaults to `nothing` (no logging).
* `log_options`: a `LogOptions` struct controlling logging behavior such as
  `log_frequency`. Separate from `logger` to allow configuring logging behavior
  independently of the logging backend.
* `iteration`: a `Ref{Int}` or `Vector{Int}` for controlling the iteration counter.
  If `nothing` (default), an internal counter is used with self-incrementing enabled.
* `kwargs`: Extra keyword arguments which are splatted to the `OptimizationProblem` on
  `solve`.
"""
@concrete struct PhysicsInformedNN <: AbstractPINN
    chain <: Union{AbstractLuxLayer, AbstractArray{<:AbstractLuxLayer}}
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    init_params
    init_states
    phi <: Union{Phi, AbstractArray{<:Phi}}
    derivative
    param_estim
    additional_loss
    adaptive_loss
    logger
    log_options::LogOptions
    iteration
    self_increment::Bool
    multioutput::Bool
    kwargs
end

function PhysicsInformedNN(
        chain, strategy; init_params = nothing, init_states = nothing, derivative = nothing,
        param_estim = false, phi::Union{Nothing, Phi, AbstractArray{<:Phi}} = nothing,
        additional_loss = nothing, adaptive_loss = nothing, logger = nothing,
        log_options = LogOptions(), iteration = nothing, kwargs...
    )
    multioutput = chain isa AbstractArray
    if multioutput
        chain = map(chain) do cᵢ
            cᵢ isa AbstractLuxLayer && return cᵢ
            return FromFluxAdaptor()(cᵢ)
        end
    else
        chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    end

    phi = phi === nothing ?
        (
            multioutput ?
            (
                init_states === nothing ?
                map(x -> Phi(x; init_states), chain) :
                map(x -> Phi(x[1]; init_states = x[2]), zip(chain, init_states))
            ) :
            Phi(chain; init_states)
        ) :
        phi

    derivative = ifelse(derivative === nothing, numeric_derivative, derivative)

    if iteration isa Vector{Int}
        @assert length(iteration) == 1
        iteration = Ref(iteration, 1)
        self_increment = false
    elseif iteration isa Ref
        self_increment = false
    else
        iteration = Ref(1)
        self_increment = true
    end

    return PhysicsInformedNN(
        chain, strategy, init_params, init_states, phi, derivative,
        param_estim, additional_loss, adaptive_loss, logger, log_options, iteration,
        self_increment, multioutput, kwargs
    )
end

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

"""
    AbstractAdaptiveLoss

An abstract type for adaptive loss weighting strategies used in PINN training.
Concrete subtypes dynamically adjust the relative importance of different loss
components (PDE, boundary conditions, additional) during optimization.

See also: [`NonAdaptiveLoss`](@ref), [`GradientScaleAdaptiveLoss`](@ref),
[`MiniMaxAdaptiveLoss`](@ref).
"""
abstract type AbstractAdaptiveLoss end

"""
    PINNRepresentation

An internal representation of a physics-informed neural network (PINN). This is the struct
used internally and returned for introspection by `symbolic_discretize`.

## Fields

- `eqs`: The equations of the PDE.
- `bcs`: The boundary condition equations.
- `domains`: The domains for each of the independent variables.
- `eq_params`: The symbolic parameters of the PDE system, or `SciMLBase.NullParameters()` if none.
- `defaults`: The default values (initial conditions) dictionary from the PDESystem.
- `default_p`: The numeric values of the PDE parameters extracted from `defaults`, or `nothing`
  if there are no parameters.
- `param_estim`: Whether parameters of the differential equation are estimated alongside the
  neural network weights.
- `additional_loss`: The `additional_loss` function as provided by the user.
- `adaloss`: The adaptive loss function, an `AbstractAdaptiveLoss`.
- `depvars`: The dependent variables of the system as a `Vector{Symbol}`.
- `indvars`: The independent variables of the system as a `Vector{Symbol}`.
- `dict_indvars`: A `Dict{Symbol, Int}` mapping independent variable symbols to their integer indices.
- `dict_depvars`: A `Dict{Symbol, Int}` mapping dependent variable symbols to their integer indices.
- `dict_depvar_input`: A `Dict{Symbol, Vector{Symbol}}` mapping each dependent variable symbol to
  the vector of independent variable symbols that it depends on.
- `logger`: The logger as provided by the user (e.g. from TensorBoardLogger.jl), or `nothing`.
- `multioutput`: Whether there are multiple outputs, i.e. a system of PDEs.
- `iteration`: The iteration counter used inside the cost function.
- `init_params`: The initial parameters as provided by the user. If the PDE is a system of PDEs, this
  will be an array of arrays. If Lux.jl is used, then this is an array of ComponentArrays.
- `flat_init_params`: The initial parameters as a flattened array. This is the array that is used in the
  construction of the OptimizationProblem. If a Lux.jl neural network is used, then this
  flattened form is a `ComponentArray`. If the equation is a system of equations, then
  `flat_init_params.depvar.x` are the parameters for the neural network corresponding
  to the dependent variable `x`, and i.e. if `depvar[i] == :x` then for `phi[i]`.
  If `param_estim = true`, then `flat_init_params.p` are the parameters and
  `flat_init_params.depvar.x` are the neural network parameters, so
  `flat_init_params.depvar.x` would be the parameters of the neural network for the
  dependent variable `x` if it's a system.
- `phi`: The representation of the test function of the PDE solution.
- `derivative`: The function used for computing the derivative.
- `strategy`: The training strategy as provided by the user, an `AbstractTrainingStrategy`.
- `pde_indvars`: The independent variables used in each PDE equation. For `QuadratureTraining`,
  these are the arguments; for other strategies, these are the variables.
- `bc_indvars`: The independent variables used in each boundary condition. For `QuadratureTraining`,
  these are the arguments; for other strategies, these are the variables.
- `pde_integration_vars`: The integration variables found in each PDE equation (for integral terms).
- `bc_integration_vars`: The integration variables found in each boundary condition (for integral terms).
- `integral`: The numeric integration function used for integral terms in the PDE, or `nothing`
  before initialization.
- `symbolic_pde_loss_functions`: The PDE loss functions as represented in Julia AST.
- `symbolic_bc_loss_functions`: The boundary condition loss functions as represented in Julia AST.
- `loss_functions`: The `PINNLossFunctions`, i.e. the generated loss functions.
"""
mutable struct PINNRepresentation
    eqs::Any
    bcs::Any
    domains::Any
    eq_params::Any
    defaults::Any
    default_p::Any
    param_estim::Bool
    additional_loss::Any
    adaloss::AbstractAdaptiveLoss
    depvars::Vector{Symbol}
    indvars::Vector{Symbol}
    dict_indvars::Dict{Symbol, Int}
    dict_depvars::Dict{Symbol, Int}
    dict_depvar_input::Dict{Symbol, Vector{Symbol}}
    logger::Any
    multioutput::Bool
    iteration::Any
    init_params::Any
    flat_init_params::Any
    phi::Any
    derivative::Any
    strategy::AbstractTrainingStrategy
    pde_indvars::Any
    bc_indvars::Any
    pde_integration_vars::Any
    bc_integration_vars::Any
    integral::Any
    symbolic_pde_loss_functions::Any
    symbolic_bc_loss_functions::Any
    loss_functions::Any
end

"""
    PINNLossFunctions

The generated functions from the `PINNRepresentation`.

## Fields

- `bc_loss_functions`: The boundary condition loss functions.
- `pde_loss_functions`: The PDE loss functions.
- `full_loss_function`: The full loss function, combining the PDE and boundary condition loss
  functions. This is the loss function that is used by the optimizer.
- `additional_loss_function`: The wrapped `additional_loss`, as pieced together for the optimizer.
- `datafree_pde_loss_functions`: The pre-data version of the PDE loss function.
- `datafree_bc_loss_functions`: The pre-data version of the BC loss function.
"""
@concrete struct PINNLossFunctions
    bc_loss_functions
    pde_loss_functions
    full_loss_function
    additional_loss_function
    datafree_pde_loss_functions
    datafree_bc_loss_functions
end

get_u() = (cord, θ, phi) -> phi(cord, θ)

# the method to calculate the derivative
function numeric_derivative(phi, u, x, εs, order, θ)
    ε = εs[order]
    _epsilon = inv(first(ε[ε .!= zero(ε)]))
    ε = ε |> safe_get_device(x)

    # any(x->x!=εs[1],εs)
    # εs is the epsilon for each order, if they are all the same then we use a fancy formula
    # if order 1, this is trivially true

    if order > 4 || any(x -> x != εs[1], εs)
        return (
            numeric_derivative(phi, u, x .+ ε, @view(εs[1:(end - 1)]), order - 1, θ)
                .-
                numeric_derivative(phi, u, x .- ε, @view(εs[1:(end - 1)]), order - 1, θ)
        ) .*
            _epsilon ./ 2
    elseif order == 4
        return (
            u(x .+ 2 .* ε, θ, phi) .- 4 .* u(x .+ ε, θ, phi)
                .+
                6 .* u(x, θ, phi)
                .-
                4 .* u(x .- ε, θ, phi) .+ u(x .- 2 .* ε, θ, phi)
        ) .* _epsilon^4
    elseif order == 3
        return (
            u(x .+ 2 .* ε, θ, phi) .- 2 .* u(x .+ ε, θ, phi) .+ 2 .* u(x .- ε, θ, phi)
                -
                u(x .- 2 .* ε, θ, phi)
        ) .* _epsilon^3 ./ 2
    elseif order == 2
        return (u(x .+ ε, θ, phi) .+ u(x .- ε, θ, phi) .- 2 .* u(x, θ, phi)) .* _epsilon^2
    elseif order == 1
        return (u(x .+ ε, θ, phi) .- u(x .- ε, θ, phi)) .* _epsilon ./ 2
    else
        error("This shouldn't happen!")
    end
end
