"""
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

"""This function is defined here as stubs to be overriden by the subpackage NeuralPDELogging if imported"""
function logvector(logger, v::AbstractVector{R}, name::AbstractString,
                   step::Integer) where {R <: Real}
    nothing
end

"""This function is defined here as stubs to be overriden by the subpackage NeuralPDELogging if imported"""
function logscalar(logger, s::R, name::AbstractString, step::Integer) where {R <: Real}
    nothing
end

abstract type AbstractPINN end

"""
Algorithm for solving Physics-Informed Neural Networks problems.

Arguments:
* `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `strategy`: determines which training strategy will be used,
* `init_params`: the initial parameter of the neural network,
* `phi`: a trial solution,
* `derivative`: method that calculates the derivative.
"""
struct PhysicsInformedNN{T, P, PH, DER, PE, AL, ADA, LOG, K} <: AbstractPINN
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
                                           kwargs...) where {iip}
        if init_params === nothing
            if chain isa AbstractArray
                initθ = DiffEqFlux.initial_params.(chain)
            else
                initθ = DiffEqFlux.initial_params(chain)
            end
        else
            initθ = init_params
        end

        multioutput = typeof(chain) <: AbstractArray

        type_initθ = multioutput ? Base.promote_typeof.(initθ)[1] :
                     Base.promote_typeof(initθ)

        if phi === nothing
            if multioutput
                _phi = Phi.(chain)
            else
                _phi = Phi(chain)
            end
        else
            _phi = phi
        end

        if derivative === nothing
            _derivative = numeric_derivative
        else
            _derivative = derivative
        end

        if !(typeof(adaptive_loss) <: AbstractAdaptiveLoss)
            floattype = eltype(initθ)
            if floattype <: Vector
                floattype = eltype(floattype)
            end
            adaptive_loss = NonAdaptiveLoss{floattype}()
        end

        if iteration isa Vector{Int64}
            self_increment = false
        else
            iteration = [1]
            self_increment = true
        end

        new{typeof(strategy), typeof(initθ), typeof(_phi), typeof(_derivative),
            typeof(param_estim),
            typeof(additional_loss), typeof(adaptive_loss), typeof(logger), typeof(kwargs)}(strategy,
                                                                                            initθ,
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

mutable struct PINNRepresentation
    eqs::Any
    bcs::Any
    domains::Any
    eq_params::Any
    defaults::Any
    default_p::Any
    param_estim::Any
    additional_loss::Any
    adaloss::Any
    depvars::Any
    indvars::Any
    dict_indvars::Any
    dict_depvars::Any
    dict_depvar_input::Any
    multioutput::Bool
    initθ::Any
    flat_initθ::Any
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

struct PINNLossFunctions
    bc_loss_functions::Any
    pde_loss_functions::Any
    full_loss_function::Any
    additional_loss_function::Any
    inner_pde_loss_functions::Any
    inner_bc_loss_functions::Any
end

"""
An encoding of the test function phi that is used for calculating the PDE
value at domain points x

Fields:

- `f`: A representation of the chain function. If FastChain, then `f(x,p)`,
  if Chain then `f(p)(x)` (from Flux.destructure)
"""
struct Phi{C}
    f::C
    Phi(chain::FastChain) = new{typeof(chain)}(chain)
    Phi(chain::Flux.Chain) = (re = Flux.destructure(chain)[2]; new{typeof(re)}(re))
end

(f::Phi{<:FastChain})(x, θ) = f.f(adapt(parameterless_type(θ), x), θ)
(f::Phi{<:Optimisers.Restructure})(x, θ) = f.f(θ)(adapt(parameterless_type(θ), x))

function get_u()
    u = (cord, θ, phi) -> phi(cord, θ)
end

# the method to calculate the derivative
function numeric_derivative(phi, u, x, εs, order, θ)
    _epsilon = one(eltype(θ)) / (2 * cbrt(eps(eltype(θ))))
    ε = εs[order]
    ε = adapt(parameterless_type(θ), ε)
    x = adapt(parameterless_type(θ), x)
    if order > 1
        return (numeric_derivative(phi, u, x .+ ε, εs, order - 1, θ)
                .-
                numeric_derivative(phi, u, x .- ε, εs, order - 1, θ)) .* _epsilon
    else
        return (u(x .+ ε, θ, phi) .- u(x .- ε, θ, phi)) .* _epsilon
    end
end
