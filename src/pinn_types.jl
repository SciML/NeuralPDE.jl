"""
???
"""
struct LogOptions
    log_frequency::Int64
    # TODO: add in an option for saving plots in the log. this is currently not done because the type of plot is dependent on the PDESystem
    #       possible solution: pass in a plot function?
    #       this is somewhat important because we want to support plotting adaptive weights that depend on pde independent variables
    #       and not just one weight for each loss function, i.e. pde_loss_weights(i, t, x) and since this would be function-internal,
    #       we'd want the plot & log to happen internally as well
    #       plots of the learned function can happen in the outer callback, but we might want to offer that here too

    SciMLBase.@add_kwonly function LogOptions(; log_frequency = 50)
        new(convert(Int64, log_frequency))
    end
end

"""This function is defined here as stubs to be overridden by the subpackage NeuralPDELogging if imported"""
function logvector(logger, v::AbstractVector{R}, name::AbstractString,
        step::Integer) where {R <: Real}
    nothing
end

"""This function is defined here as stubs to be overridden by the subpackage NeuralPDELogging if imported"""
function logscalar(logger, s::R, name::AbstractString, step::Integer) where {R <: Real}
    nothing
end

"""
    PhysicsInformedNN(chain,
                    strategy;
                    init_params = nothing,
                    phi = nothing,
                    param_estim = false,
                    additional_loss = nothing,
                    adaptive_loss = nothing,
                    logger = nothing,
                    log_options = LogOptions(),
                    iteration = nothing,
                    kwargs...)

A `discretize` algorithm for the ModelingToolkit PDESystem interface, which transforms a
`PDESystem` into an `OptimizationProblem` using the Physics-Informed Neural Networks (PINN)
methodology.

## Positional Arguments

* `chain`: a vector of Lux/Flux chains with a d-dimensional input and a
           1-dimensional output corresponding to each of the dependent variables. Note that this
           specification respects the order of the dependent variables as specified in the PDESystem.
           Flux chains will be converted to Lux internally using `adapt(FromFluxAdaptor(false, false), chain)`.
* `strategy`: determines which training strategy will be used. See the Training Strategy
              documentation for more details.

## Keyword Arguments

* `init_params`: the initial parameters of the neural networks. If `init_params` is not
  given, then the neural network default parameters are used. Note that for Lux, the default
  will convert to Float64.
* `phi`: a trial solution, specified as `phi(x,p)` where `x` is the coordinates vector for
  the dependent variable and `p` are the weights of the phi function (generally the weights
  of the neural network defining `phi`). By default, this is generated from the `chain`. This
  should only be used to more directly impose functional information in the training problem,
  for example imposing the boundary condition by the test function formulation.
* `adaptive_loss`: the choice for the adaptive loss function. See the
  [adaptive loss page](@ref adaptive_loss) for more details. Defaults to no adaptivity.
* `additional_loss`: a function `additional_loss(phi, θ, p_)` where `phi` are the neural
  network trial solutions, `θ` are the weights of the neural network(s), and `p_` are the
  hyperparameters of the `OptimizationProblem`. If `param_estim = true`, then `θ` additionally
  contains the parameters of the differential equation appended to the end of the vector.
* `param_estim`: whether the parameters of the differential equation should be included in
  the values sent to the `additional_loss` function. Defaults to `false`.
* `logger`: ?? needs docs
* `log_options`: ?? why is this separate from the logger?
* `iteration`: used to control the iteration counter???
* `kwargs`: Extra keyword arguments which are splatted to the `OptimizationProblem` on `solve`.
"""
struct PhysicsInformedNN{T, P, PH, DER, PE, AL, ADA, LOG, K} <: AbstractPINN
    chain::Any
    strategy::T
    init_params::P
    phi::PH
    derivative::DER
    param_estim::PE
    additional_loss::AL
    adaptive_loss::ADA
    logger::LOG
    log_options::LogOptions
    iteration::Vector{Int64}
    self_increment::Bool
    multioutput::Bool
    kwargs::K

    @add_kwonly function PhysicsInformedNN(chain,
            strategy;
            init_params = nothing,
            phi = nothing,
            derivative = nothing,
            param_estim = false,
            additional_loss = nothing,
            adaptive_loss = nothing,
            logger = nothing,
            log_options = LogOptions(),
            iteration = nothing,
            kwargs...)
        multioutput = chain isa AbstractArray
        if multioutput
            !all(i -> i isa Lux.AbstractLuxLayer, chain) &&
                (chain = Lux.transform.(chain))
        else
            !(chain isa Lux.AbstractLuxLayer) &&
                (chain = adapt(FromFluxAdaptor(false, false), chain))
        end
        if phi === nothing
            if multioutput
                _phi = Phi.(chain)
            else
                _phi = Phi(chain)
            end
        else
            if multioutput
                all([phi.f[i] isa Lux.AbstractLuxLayer for i in eachindex(phi.f)]) ||
                    throw(ArgumentError("Only Lux Chains are supported"))
            else
                (phi.f isa Lux.AbstractLuxLayer) ||
                    throw(ArgumentError("Only Lux Chains are supported"))
            end
            _phi = phi
        end

        if derivative === nothing
            _derivative = numeric_derivative
        else
            _derivative = derivative
        end

        if iteration isa Vector{Int64}
            self_increment = false
        else
            iteration = [1]
            self_increment = true
        end

        new{typeof(strategy), typeof(init_params), typeof(_phi), typeof(_derivative),
            typeof(param_estim),
            typeof(additional_loss), typeof(adaptive_loss), typeof(logger), typeof(kwargs)}(
            chain,
            strategy,
            init_params,
            _phi,
            _derivative,
            param_estim,
            additional_loss,
            adaptive_loss,
            logger,
            log_options,
            iteration,
            self_increment,
            multioutput,
            kwargs)
    end
end

"""
    BayesianPINN(chain,
                  strategy;
                  init_params = nothing,
                  phi = nothing,
                  param_estim = false,
                  additional_loss = nothing,
                  adaptive_loss = nothing,
                  logger = nothing,
                  log_options = LogOptions(),
                  iteration = nothing,
                  dataset = nothing,
                  kwargs...)

A `discretize` algorithm for the ModelingToolkit PDESystem interface, which transforms a
`PDESystem` into a likelihood function used for HMC based Posterior Sampling Algorithms [AdvancedHMC.jl](https://turinglang.org/AdvancedHMC.jl/stable/)
which is later optimized upon to give the Solution Distribution of the PDE, using the Physics-Informed Neural Networks (PINN)
methodology.

## Positional Arguments

* `chain`: a vector of Lux.jl chains with a d-dimensional input and a
  1-dimensional output corresponding to each of the dependent variables. Note that this
  specification respects the order of the dependent variables as specified in the PDESystem.
* `strategy`: determines which training strategy will be used. See the Training Strategy
  documentation for more details.

## Keyword Arguments

* `Dataset`: A vector of matrix, each matrix for ith dependant
  variable and first col in matrix is for dependant variables,
  remaining columns for independent variables. Needed for inverse problem solving.
* `init_params`: the initial parameters of the neural networks. If `init_params` is not
  given, then the neural network default parameters are used. Note that for Lux, the default
  will convert to Float64.
* `phi`: a trial solution, specified as `phi(x,p)` where `x` is the coordinates vector for
  the dependent variable and `p` are the weights of the phi function (generally the weights
  of the neural network defining `phi`). By default, this is generated from the `chain`. This
  should only be used to more directly impose functional information in the training problem,
  for example imposing the boundary condition by the test function formulation.
* `adaptive_loss`: (STILL WIP), the choice for the adaptive loss function. See the
  [adaptive loss page](@ref adaptive_loss) for more details. Defaults to no adaptivity.
* `additional_loss`: a function `additional_loss(phi, θ, p_)` where `phi` are the neural
  network trial solutions, `θ` are the weights of the neural network(s), and `p_` are the
  hyperparameters . If `param_estim = true`, then `θ` additionally
  contains the parameters of the differential equation appended to the end of the vector.
* `param_estim`: whether the parameters of the differential equation should be included in
  the values sent to the `additional_loss` function. Defaults to `false`.
* `logger`: ?? needs docs
* `log_options`: ?? why is this separate from the logger?
* `iteration`: used to control the iteration counter???
* `kwargs`: Extra keyword arguments.
"""
struct BayesianPINN{T, P, PH, DER, PE, AL, ADA, LOG, D, K} <: AbstractPINN
    chain::Any
    strategy::T
    init_params::P
    phi::PH
    derivative::DER
    param_estim::PE
    additional_loss::AL
    adaptive_loss::ADA
    logger::LOG
    log_options::LogOptions
    iteration::Vector{Int64}
    self_increment::Bool
    multioutput::Bool
    dataset::D
    kwargs::K

    @add_kwonly function BayesianPINN(chain,
            strategy;
            init_params = nothing,
            phi = nothing,
            derivative = nothing,
            param_estim = false,
            additional_loss = nothing,
            adaptive_loss = nothing,
            logger = nothing,
            log_options = LogOptions(),
            iteration = nothing,
            dataset = nothing,
            kwargs...)
        multioutput = chain isa AbstractArray
        if multioutput
            !all(i -> i isa Lux.AbstractLuxLayer, chain) &&
                (chain = Lux.transform.(chain))
        else
            !(chain isa Lux.AbstractLuxLayer) &&
                (chain = adapt(FromFluxAdaptor(false, false), chain))
        end
        if phi === nothing
            if multioutput
                _phi = Phi.(chain)
            else
                _phi = Phi(chain)
            end
        else
            if multioutput
                all([phi.f[i] isa Lux.AbstractLuxLayer for i in eachindex(phi.f)]) ||
                    throw(ArgumentError("Only Lux Chains are supported"))
            else
                (phi.f isa Lux.AbstractLuxLayer) ||
                    throw(ArgumentError("Only Lux Chains are supported"))
            end
            _phi = phi
        end

        if derivative === nothing
            _derivative = numeric_derivative
        else
            _derivative = derivative
        end

        if iteration isa Vector{Int64}
            self_increment = false
        else
            iteration = [1]
            self_increment = true
        end

        if dataset isa Nothing
            dataset = (nothing, nothing)
        end

        new{typeof(strategy), typeof(init_params), typeof(_phi), typeof(_derivative),
            typeof(param_estim),
            typeof(additional_loss), typeof(adaptive_loss), typeof(logger), typeof(dataset),
            typeof(kwargs)}(chain,
            strategy,
            init_params,
            _phi,
            _derivative,
            param_estim,
            additional_loss,
            adaptive_loss,
            logger,
            log_options,
            iteration,
            self_increment,
            multioutput,
            dataset,
            kwargs)
    end
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
    iteration::Vector{Int}
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

"""
An encoding of the test function phi that is used for calculating the PDE
value at domain points x

Fields:

- `f`: A representation of the chain function.
- `st`: The state of the Lux.AbstractLuxLayer. It should be updated on each call.
"""
mutable struct Phi{C, S}
    f::C
    st::S
    function Phi(chain::Lux.AbstractLuxLayer)
        st = Lux.initialstates(Random.default_rng(), chain)
        new{typeof(chain), typeof(st)}(chain, st)
    end
end

function (f::Phi{<:Lux.AbstractLuxLayer})(x::Number, θ)
    y, st = f.f(adapt(parameterless_type(ComponentArrays.getdata(θ)), [x]), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    y
end

function (f::Phi{<:Lux.AbstractLuxLayer})(x::AbstractArray, θ)
    y, st = f.f(adapt(parameterless_type(ComponentArrays.getdata(θ)), x), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    y
end

function get_u()
    u = (cord, θ, phi) -> phi(cord, θ)
end

# the method to calculate the derivative
function numeric_derivative(phi, u, x, εs, order, θ)
    _type = parameterless_type(ComponentArrays.getdata(θ))

    ε = εs[order]
    _epsilon = inv(first(ε[ε .!= zero(ε)]))

    ε = adapt(_type, ε)
    x = adapt(_type, x)

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
