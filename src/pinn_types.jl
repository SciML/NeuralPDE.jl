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
                      iteration = nothing, symbolic_parser = false, kwargs...)

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
* `logger`: a logging object (e.g. a TensorBoardLogger) used for recording loss values and
  adaptive weights during training. Defaults to `nothing` (no logging).
* `log_options`: a `LogOptions` struct controlling logging frequency (e.g. how often to write
  loss values to the logger). Separate from `logger` to allow configuring log frequency
  independently of the logger type.
* `iteration`: an optional external iteration counter (a `Ref{Int}` or `Vector{Int}` of
  length 1) shared with the caller so the caller can read or control the training step count.
  If not provided, an internal counter is created and auto-incremented.
* `symbolic_parser`: whether to use the experimental symbolic Prewalk-based PINN parser
  instead of the default RuntimeGeneratedFunction-based parser. When `true`, equation parsing
  and compilation are routed through the symbolic parser, which uses ForwardDiff for
  derivatives. Defaults to `false`.
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
    symbolic_parser::Bool
    kwargs
end

function PhysicsInformedNN(
        chain, strategy; init_params = nothing, init_states = nothing, derivative = nothing,
        param_estim = false, phi::Union{Nothing, Phi, AbstractArray{<:Phi}} = nothing,
        additional_loss = nothing, adaptive_loss = nothing, logger = nothing,
        log_options = LogOptions(), iteration = nothing, symbolic_parser::Bool = false, kwargs...
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
        self_increment, multioutput, symbolic_parser, kwargs
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
    The symbolic parameters of the PDE system (e.g. physical constants to be estimated).
    Corresponds to `pde_system.ps`. Set to `SciMLBase.NullParameters()` when there are no
    parameters.
    """
    eq_params::Any
    """
    The default values of PDE parameters as a dictionary mapping each parameter symbol to its
    numeric value. Corresponds to `pde_system.initial_conditions`.
    """
    defaults::Any
    """
    The numeric default values of the PDE parameters as a plain `Vector`, extracted from
    `defaults` for use inside the loss function. `nothing` when `eq_params` is
    `NullParameters`.
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
    A `Dict{Symbol, Int}` mapping each independent variable name (e.g. `:x`, `:t`) to its
    positional index in the coordinate vector. Used to build collocation point expressions.
    """
    dict_indvars::Any
    """
    A `Dict{Symbol, Int}` mapping each dependent variable name (e.g. `:u`, `:v`) to its
    positional index among the outputs. Used to index into `phi` and `θ` for multi-output
    systems.
    """
    dict_depvars::Any
    """
    A `Dict{Symbol, Vector{Symbol}}` mapping each dependent variable name to the list of
    independent variable names it depends on. For example, `u(x, t)` maps `:u => [:x, :t]`.
    Used to build the correct coordinate slices for each network input.
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
    For each PDE equation, the list of independent variables that appear in it. Used to build
    the correct collocation point layout for each loss term. For `QuadratureTraining` this
    holds the full argument list; for other strategies it holds only the variable symbols.
    """
    pde_indvars::Any
    """
    For each boundary condition equation, the list of independent variables that appear in
    it. Analogous to `pde_indvars` but for boundary loss terms.
    """
    bc_indvars::Any
    """
    For each PDE equation, the list of independent variables that are being integrated over
    (non-empty only when the equation contains a `Symbolics.Integral` term).
    """
    pde_integration_vars::Any
    """
    For each boundary condition equation, the list of independent variables that are being
    integrated over (non-empty only when the BC contains a `Symbolics.Integral` term).
    """
    bc_integration_vars::Any
    """
    The compiled numeric integral function, built by `get_numeric_integral`. Evaluates
    `Symbolics.Integral` terms at runtime using `Integrals.jl` quadrature.
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
