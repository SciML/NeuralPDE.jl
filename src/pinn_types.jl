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

"""This function is defined here as stubs to be overridden by the subpackage NeuralPDELogging if imported"""
function logvector(logger, v::AbstractVector{<:Real}, name::AbstractString, step::Integer)
    nothing
end

"""This function is defined here as stubs to be overridden by the subpackage NeuralPDELogging if imported"""
function logscalar(logger, s::Real, name::AbstractString, step::Integer)
    nothing
end

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

function Phi(layer::AbstractLuxLayer)
    return Phi(StatefulLuxLayer{true}(
        layer, nothing, initialstates(Random.default_rng(), layer)))
end

(f::Phi)(x::Number, θ) = only(cdev(f([x], θ)))

(f::Phi)(x::AbstractArray, θ) = f.smodel(safe_get_device(θ)(x), θ)

"""
    PhysicsInformedNN(chain, strategy; init_params = nothing, phi = nothing,
                      param_estim = false, additional_loss = nothing,
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
* `logger`: ?? needs docs
* `log_options`: ?? why is this separate from the logger?
* `iteration`: used to control the iteration counter???
* `kwargs`: Extra keyword arguments which are splatted to the `OptimizationProblem` on
  `solve`.
"""
@concrete struct PhysicsInformedNN <: AbstractPINN
    chain <: Union{AbstractLuxLayer, AbstractArray{<:AbstractLuxLayer}}
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    init_params
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
        chain, strategy; init_params = nothing, derivative = nothing, param_estim = false,
        phi::Union{Nothing, Phi, AbstractArray{<:Phi}} = nothing, additional_loss = nothing,
        adaptive_loss = nothing, logger = nothing, log_options = LogOptions(),
        iteration = nothing, kwargs...)
    multioutput = chain isa AbstractArray
    if multioutput
        chain = map(chain) do cᵢ
            cᵢ isa AbstractLuxLayer && return cᵢ
            return FromFluxAdaptor()(cᵢ)
        end
    else
        chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    end

    phi = phi === nothing ? (multioutput ? map(Phi, chain) : Phi(chain)) : phi

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

    return PhysicsInformedNN(chain, strategy, init_params, phi, derivative, param_estim,
        additional_loss, adaptive_loss, logger, log_options, iteration, self_increment,
        multioutput, kwargs)
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
`PINNRepresentation``

An internal representation of a physics-informed neural network (PINN). This is the struct
used internally and returned for introspection by `symbolic_discretize`.

## Fields

$(FIELDS)
"""
mutable struct PINNRepresentation
    """
    The equations of the PDE
    """
    eqs::Any
    """
    The boundary condition equations
    """
    bcs::Any
    """
    The domains for each of the independent variables
    """
    domains::Any
    """
    ???
    """
    eq_params::Any
    """
    ???
    """
    defaults::Any
    """
    ???
    """
    default_p::Any
    """
    Whether parameters are to be appended to the `additional_loss`
    """
    param_estim::Any
    """
    The `additional_loss` function as provided by the user
    """
    additional_loss::Any
    """
    The adaptive loss function
    """
    adaloss::Any
    """
    The dependent variables of the system
    """
    depvars::Any
    """
    The independent variables of the system
    """
    indvars::Any
    """
    A dictionary form of the independent variables. Define the structure ???
    """
    dict_indvars::Any
    """
    A dictionary form of the dependent variables. Define the structure ???
    """
    dict_depvars::Any
    """
    ???
    """
    dict_depvar_input::Any
    """
    The logger as provided by the user
    """
    logger::Any
    """
    Whether there are multiple outputs, i.e. a system of PDEs
    """
    multioutput::Bool
    """
    The iteration counter used inside the cost function
    """
    iteration::Any
    """
    The initial parameters as provided by the user. If the PDE is a system of PDEs, this
    will be an array of arrays. If Lux.jl is used, then this is an array of ComponentArrays.
    """
    init_params::Any
    """
    The initial parameters as a flattened array. This is the array that is used in the
    construction of the OptimizationProblem. If a Lux.jl neural network is used, then this
    flattened form is a `ComponentArray`. If the equation is a system of equations, then
    `flat_init_params.depvar.x` are the parameters for the neural network corresponding
    to the dependent variable `x`, and i.e. if `depvar[i] == :x` then for `phi[i]`.
    If `param_estim = true`, then `flat_init_params.p` are the parameters and
    `flat_init_params.depvar.x` are the neural network parameters, so
    `flat_init_params.depvar.x` would be the parameters of the neural network for the
    dependent variable `x` if it's a system.
    """
    flat_init_params::Any
    """
    The representation of the test function of the PDE solution
    """
    phi::Any
    """
    The function used for computing the derivative
    """
    derivative::Any
    """
    The training strategy as provided by the user
    """
    strategy::AbstractTrainingStrategy
    """
    ???
    """
    pde_indvars::Any
    """
    ???
    """
    bc_indvars::Any
    """
    ???
    """
    pde_integration_vars::Any
    """
    ???
    """
    bc_integration_vars::Any
    """
    ???
    """
    integral::Any
    """
    The PDE loss functions as represented in Julia AST
    """
    symbolic_pde_loss_functions::Any
    """
    The boundary condition loss functions as represented in Julia AST
    """
    symbolic_bc_loss_functions::Any
    """
    The PINNLossFunctions, i.e. the generated loss functions
    """
    loss_functions::Any
end

"""
`PINNLossFunctions``

The generated functions from the PINNRepresentation

## Fields

$(FIELDS)
"""
struct PINNLossFunctions
    """
    The boundary condition loss functions
    """
    bc_loss_functions::Any
    """
    The PDE loss functions
    """
    pde_loss_functions::Any
    """
    The full loss function, combining the PDE and boundary condition loss functions.
    This is the loss function that is used by the optimizer.
    """
    full_loss_function::Any
    """
    The wrapped `additional_loss`, as pieced together for the optimizer.
    """
    additional_loss_function::Any
    """
    The pre-data version of the PDE loss function
    """
    datafree_pde_loss_functions::Any
    """
    The pre-data version of the BC loss function
    """
    datafree_bc_loss_functions::Any
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
        return (numeric_derivative(phi, u, x .+ ε, @view(εs[1:(end - 1)]), order - 1, θ)
                .-
                numeric_derivative(phi, u, x .- ε, @view(εs[1:(end - 1)]), order - 1, θ)) .*
               _epsilon ./ 2
    elseif order == 4
        return (u(x .+ 2 .* ε, θ, phi) .- 4 .* u(x .+ ε, θ, phi)
                .+
                6 .* u(x, θ, phi)
                .-
                4 .* u(x .- ε, θ, phi) .+ u(x .- 2 .* ε, θ, phi)) .* _epsilon^4
    elseif order == 3
        return (u(x .+ 2 .* ε, θ, phi) .- 2 .* u(x .+ ε, θ, phi) .+ 2 .* u(x .- ε, θ, phi)
                -
                u(x .- 2 .* ε, θ, phi)) .* _epsilon^3 ./ 2
    elseif order == 2
        return (u(x .+ ε, θ, phi) .+ u(x .- ε, θ, phi) .- 2 .* u(x, θ, phi)) .* _epsilon^2
    elseif order == 1
        return (u(x .+ ε, θ, phi) .- u(x .- ε, θ, phi)) .* _epsilon ./ 2
    else
        error("This shouldn't happen!")
    end
end
