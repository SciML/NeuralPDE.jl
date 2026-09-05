# Native symbolic-array lowering for ModelingToolkit OptimizationSystem PINNs.

struct PINNArrayLoweringError <: Exception
    message::String
end

Base.showerror(io::IO, err::PINNArrayLoweringError) = print(io, err.message)

function _constant_coordinate_on_theta_device(cord, theta)
    return @ignore_derivatives safe_get_device(theta)(collect(cord))
end

_runtime_theta(theta, _) = theta
_runtime_theta(theta::AbstractVector, theta_axes) = ComponentArray(theta, theta_axes)
_runtime_theta(theta::ComponentArray, _) = theta

struct PINNPhiArrayOperator{P, A}
    phi::P
    theta_axes::A
    dependent_variable_index::Int
    multioutput::Bool
end

Base.nameof(::PINNPhiArrayOperator) = :pinn_phi_array

function (op::PINNPhiArrayOperator)(cord::AbstractMatrix, theta)
    structured_theta = _runtime_theta(theta, op.theta_axes)
    cord = _constant_coordinate_on_theta_device(cord, structured_theta)
    return phi_eval(
        op.phi, cord, structured_theta, op.dependent_variable_index, op.multioutput
    )
end

@register_array_symbolic (op::PINNPhiArrayOperator)(
    cord::AbstractMatrix, theta::AbstractVector
) begin
    size = (1, size(cord, 2))
    eltype = promote_type(eltype(cord), eltype(theta))
end

struct PINNDerivativeArrayOperator{D, P, A, V}
    derivative::D
    phi::P
    theta_axes::A
    directions::V
    dependent_variable_index::Int
    multioutput::Bool
end

Base.nameof(::PINNDerivativeArrayOperator) = :pinn_derivative_array

function (op::PINNDerivativeArrayOperator)(cord::AbstractMatrix, theta)
    structured_theta = _runtime_theta(theta, op.theta_axes)
    cord = _constant_coordinate_on_theta_device(cord, structured_theta)
    return deriv_fd(
        op.derivative, op.phi, cord, structured_theta, collect(op.directions),
        op.dependent_variable_index, op.multioutput
    )
end

@register_array_symbolic (op::PINNDerivativeArrayOperator)(
    cord::AbstractMatrix, theta::AbstractVector
) begin
    size = (1, size(cord, 2))
    eltype = promote_type(eltype(cord), eltype(theta))
end

struct PINNCoordinateArrayOperator{T, I, C}
    row_types::T
    row_indices::I
    row_constants::C
end

Base.nameof(::PINNCoordinateArrayOperator) = :pinn_coordinate_array

function (op::PINNCoordinateArrayOperator)(cord::AbstractMatrix{T}, theta) where {T}
    return @ignore_derivatives begin
        rows = map(eachindex(op.row_types)) do i
            if op.row_types[i] == 1
                idx = op.row_indices[i]
                cord[idx:idx, :]
            else
                zero.(cord[1:1, :]) .+ T(op.row_constants[i])
            end
        end
        safe_get_device(theta)(collect(reduce(vcat, rows)))
    end
end

@register_array_symbolic (op::PINNCoordinateArrayOperator)(
    cord::AbstractMatrix, theta::AbstractVector
) begin
    size = (length(op.row_types), size(cord, 2))
    eltype = promote_type(eltype(cord), eltype(theta))
end

struct PINNCoordinateSliceOperator
    row_index::Int
end

Base.nameof(::PINNCoordinateSliceOperator) = :pinn_coordinate_slice

function (op::PINNCoordinateSliceOperator)(cord::AbstractMatrix, theta)
    return @ignore_derivatives safe_get_device(theta)(collect(cord[op.row_index:op.row_index, :]))
end

function ChainRulesCore.rrule(op::PINNCoordinateArrayOperator, cord, theta)
    result = op(cord, theta)
    pullback(Δ) = (
        ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
    )
    return result, pullback
end

function ChainRulesCore.rrule(op::PINNCoordinateSliceOperator, cord, theta)
    result = op(cord, theta)
    pullback(Δ) = (
        ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
    )
    return result, pullback
end

@register_array_symbolic (op::PINNCoordinateSliceOperator)(
    cord::AbstractMatrix, theta::AbstractVector
) begin
    size = (1, size(cord, 2))
    eltype = promote_type(eltype(cord), eltype(theta))
end

struct PINNRuntimeIntegrand{E, A, P, D, DI, DD, DV, CI, PI, FP, EI}
    expression::E
    theta_axes::A
    phi::P
    derivative::D
    dict_indvars::DI
    dict_depvars::DD
    dict_depvar_input::DV
    coordinate_indices::CI
    parameter_indices::PI
    fixed_parameter_values::FP
    estimated_parameter_indices::EI
    multioutput::Bool
end

struct PINNIntegralArrayOperator{I, P, D, A, V}
    integrand::I
    phi::P
    derivative::D
    theta_axes::A
    integration_variable_indices::V
    lower_values::Vector{Float64}
    upper_values::Vector{Float64}
    lower_coordinate_indices::Vector{Int}
    upper_coordinate_indices::Vector{Int}
    infinite_limits::Vector{Int}
end

Base.nameof(::PINNIntegralArrayOperator) = :pinn_integral_array

struct PINNAdditionalLossOperator{F, P, A}
    additional_loss::F
    phi::P
    theta_axes::A
    parameter_estimation::Bool
end

Base.nameof(::PINNAdditionalLossOperator) = :pinn_additional_loss

function (op::PINNAdditionalLossOperator)(theta)
    structured_theta = _runtime_theta(theta, op.theta_axes)
    network_theta, equation_parameters = op.parameter_estimation ?
        (structured_theta.depvar, structured_theta.p) : (structured_theta, nothing)
    return op.additional_loss(op.phi, network_theta, equation_parameters)
end

@register_symbolic (op::PINNAdditionalLossOperator)(theta::AbstractVector)

function (op::PINNIntegralArrayOperator)(cord::AbstractMatrix, theta)
    theta = _runtime_theta(theta, op.theta_axes)
    cord = _constant_coordinate_on_theta_device(cord, theta)
    return eval_numeric_integral(
        op.integrand, cord, op.phi, theta, op.derivative,
        collect(op.integration_variable_indices), op.lower_values, op.upper_values,
        op.lower_coordinate_indices, op.upper_coordinate_indices, op.infinite_limits
    )
end

@register_array_symbolic (op::PINNIntegralArrayOperator)(
    cord::AbstractMatrix, theta::AbstractVector
) begin
    size = (1, size(cord, 2))
    eltype = promote_type(eltype(cord), eltype(theta))
end

struct PINNSymbolicArrayContext{C, T, A, P, D, DI, DD, DV, PI, FP, FV}
    cord::C
    theta::T
    theta_axes::A
    phi::P
    derivative::D
    dict_indvars::DI
    dict_depvars::DD
    dict_depvar_input::DV
    coordinate_indices::Dict{Symbol, Int}
    parameter_indices::PI
    fixed_parameters::FP
    fixed_parameter_values::FV
    estimated_parameter_indices::Dict{Symbol, Int}
    multioutput::Bool
end

struct PINNOptimizationSystemData
    source_system::Any
    compiled_system::Any
    theta::Any
    coordinate_parameters::Any
    fixed_parameters::Any
    loss_weight_parameters::Any
    operating_point::Any
    coordinate_provider::Any
    executable::Bool
    fallback_reason::Any
end


function _optimization_training_sets(pinnrep, strategy::GridTraining, element_type)
    sets = generate_training_sets(
        pinnrep.domains, strategy.dx, pinnrep.eqs, pinnrep.bcs, element_type,
        pinnrep.dict_indvars, pinnrep.dict_depvars
    )
    return sets, nothing
end

function _representative_sample_coordinates(points::Int, bound, element_type)
    lb, ub = bound
    isempty(lb) && return _zero_dimensional_coordinates(element_type)

    midpoint = element_type.((lb .+ ub) ./ 2)
    return repeat(reshape(midpoint, :, 1), 1, points)
end

function _optimization_training_sets(pinnrep, strategy::StochasticTraining, element_type)
    pde_bounds, bc_bounds = get_bounds(
        pinnrep.domains, pinnrep.eqs, pinnrep.bcs, element_type,
        pinnrep.dict_indvars, pinnrep.dict_depvars, strategy
    )
    provider = function ()
        pde_sets = [
            generate_random_points(strategy.points, bound, element_type)
                for bound in pde_bounds
        ]
        bc_sets = [
            generate_random_points(strategy.bcs_points, bound, element_type)
                for bound in bc_bounds
        ]
        return pde_sets, bc_sets
    end
    pde_sets = [
        _representative_sample_coordinates(strategy.points, bound, element_type)
            for bound in pde_bounds
    ]
    bc_sets = [
        _representative_sample_coordinates(strategy.bcs_points, bound, element_type)
            for bound in bc_bounds
    ]
    return (pde_sets, bc_sets), provider
end

function _quasi_random_coordinates(points, bound, element_type, sampling_algorithm)
    isempty(bound[1]) && return _zero_dimensional_coordinates(element_type)
    return QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_algorithm) |>
        EltypeAdaptor{element_type}()
end

function _optimization_training_sets(pinnrep, strategy::QuasiRandomTraining, element_type)
    pde_bounds, bc_bounds = get_bounds(
        pinnrep.domains, pinnrep.eqs, pinnrep.bcs, element_type,
        pinnrep.dict_indvars, pinnrep.dict_depvars, strategy
    )

    if strategy.resampling
        provider = function ()
            pde_sets = [
                _quasi_random_coordinates(
                    strategy.points, bound, element_type, strategy.sampling_alg
                ) for bound in pde_bounds
            ]
            bc_sets = [
                _quasi_random_coordinates(
                    strategy.bcs_points, bound, element_type, strategy.sampling_alg
                ) for bound in bc_bounds
            ]
            return pde_sets, bc_sets
        end
        pde_sets = [
            _representative_sample_coordinates(strategy.points, bound, element_type)
                for bound in pde_bounds
        ]
        bc_sets = [
            _representative_sample_coordinates(strategy.bcs_points, bound, element_type)
                for bound in bc_bounds
        ]
        return (pde_sets, bc_sets), provider
    end

    strategy.minibatch > 0 || throw(
        PINNArrayLoweringError(
            "QuasiRandomTraining with resampling=false requires minibatch > 0."
        )
    )
    pde_batches = [
        isempty(bound[1]) ? fill(
            _zero_dimensional_coordinates(element_type), strategy.minibatch
        ) : generate_quasi_random_points_batch(
            strategy.points, bound, element_type, strategy.sampling_alg,
            strategy.minibatch
        ) for bound in pde_bounds
    ]
    bc_batches = [
        isempty(bound[1]) ? fill(
            _zero_dimensional_coordinates(element_type), strategy.minibatch
        ) : generate_quasi_random_points_batch(
            strategy.bcs_points, bound, element_type, strategy.sampling_alg,
            strategy.minibatch
        ) for bound in bc_bounds
    ]
    provider = () -> (
        [batch[rand(eachindex(batch))] for batch in pde_batches],
        [batch[rand(eachindex(batch))] for batch in bc_batches]
    )
    pde_sets = [first(batch) for batch in pde_batches]
    bc_sets = [first(batch) for batch in bc_batches]
    return (pde_sets, bc_sets), provider
end

function _representative_quadrature_coordinates(lb, ub, element_type, batch::Int)
    columns = max(batch, 1)
    isempty(lb) && return zeros(element_type, 1, columns)

    midpoint = element_type.((lb .+ ub) ./ 2)
    return repeat(reshape(midpoint, :, 1), 1, columns)
end

function _optimization_training_sets(pinnrep, strategy::QuadratureTraining, element_type)
    pde_bounds, bc_bounds = get_bounds(
        pinnrep.domains, pinnrep.eqs, pinnrep.bcs, element_type,
        pinnrep.dict_indvars, pinnrep.dict_depvars, strategy
    )

    pde_lbs, pde_ubs = pde_bounds
    bc_lbs, bc_ubs = bc_bounds
    pde_sets = [
        _representative_quadrature_coordinates(lb, ub, element_type, strategy.batch)
            for (lb, ub) in zip(pde_lbs, pde_ubs)
    ]
    bc_sets = [
        _representative_quadrature_coordinates(lb, ub, element_type, strategy.batch)
            for (lb, ub) in zip(bc_lbs, bc_ubs)
    ]
    return (pde_sets, bc_sets), nothing
end

function _optimization_training_sets(_, strategy::AbstractTrainingStrategy, __)
    throw(
        PINNArrayLoweringError(
            "Native OptimizationSystem lowering is not implemented for " *
            "$(typeof(strategy))."
        )
    )
end

_is_symbolic_array(x) = x isa Symbolics.Arr
_is_array_expression(x) = x isa AbstractArray || _is_symbolic_array(x)

function _symbolic_unknown_vector(name::Symbol, n::Int)
    return only(@variables $name[1:n])
end

function _symbolic_parameter_vector(name::Symbol, n::Int)
    return only(@parameters $name[1:n])
end

function _symbolic_parameter_matrix(name::Symbol, rows::Int, cols::Int)
    return only(@parameters $name[1:rows, 1:cols])
end

function _numeric_symbolic_value(value)
    unwrapped = Symbolics.value(value)
    return unwrapped isa Number ? unwrapped : nothing
end

_mtk_operating_point_value(value) = value
_mtk_operating_point_value(value::AbstractArray) = Array(value)
_mtk_operating_point_value(value::ComponentArray) = Array(getdata(value))

function (integrand::PINNRuntimeIntegrand)(cord, _, theta, __)
    context = PINNSymbolicArrayContext(
        cord, theta, integrand.theta_axes, integrand.phi, integrand.derivative,
        integrand.dict_indvars, integrand.dict_depvars, integrand.dict_depvar_input,
        integrand.coordinate_indices, integrand.parameter_indices,
        integrand.fixed_parameter_values, integrand.fixed_parameter_values,
        integrand.estimated_parameter_indices,
        integrand.multioutput
    )
    return _rebuild_array_expression(context, integrand.expression)
end

function _dependent_variable_coordinates(ctx::PINNSymbolicArrayContext, args)
    row_types, row_indices, row_constants, identity_layout = @ignore_derivatives begin
        row_types = Int[]
        row_indices = Int[]
        row_constants = Number[]

        for arg in args
            coordinate = coordinate_symbol(arg, ctx.dict_indvars)
            if coordinate !== nothing && haskey(ctx.coordinate_indices, coordinate)
                push!(row_types, 1)
                push!(row_indices, ctx.coordinate_indices[coordinate])
                push!(row_constants, 0)
                continue
            end

            value = _numeric_symbolic_value(arg)
            value === nothing && throw(
                PINNArrayLoweringError(
                    "Dependent-variable arguments must be coordinates or numeric boundary " *
                    "values in the native array lowering; encountered $(arg)."
                )
            )
            push!(row_types, 0)
            push!(row_indices, 0)
            push!(row_constants, value)
        end

        identity_layout = length(row_types) == size(ctx.cord, 1) &&
            all(==(1), row_types) && row_indices == collect(1:length(row_indices))
        Tuple(row_types), Tuple(row_indices), Tuple(row_constants), identity_layout
    end
    identity_layout && return ctx.cord

    coordinate_operator = PINNCoordinateArrayOperator(
        row_types, row_indices, row_constants
    )
    return coordinate_operator(ctx.cord, ctx.theta)
end

function _lower_array_differential(ctx::PINNSymbolicArrayContext, term)
    inner_term, differentiated_variables, dependent_variable, directions =
        @ignore_derivatives begin
            inner_term, differentiated_variables = unwrap_differentials(term)
            SymbolicUtils.iscall(inner_term) || throw(
                PINNArrayLoweringError("Only derivatives of dependent variables can be lowered.")
            )

            dependent_variable = nameof(SymbolicUtils.operation(inner_term))
            haskey(ctx.dict_depvars, dependent_variable) || throw(
                PINNArrayLoweringError(
                    "Cannot lower a derivative of the unrecognized dependent variable " *
                    "$(dependent_variable)."
                )
            )

            dependent_variable_inputs = ctx.dict_depvar_input[dependent_variable]
            directions = map(differentiated_variables) do variable
                direction = findfirst(==(variable), dependent_variable_inputs)
                direction === nothing && throw(
                    PINNArrayLoweringError(
                        "Cannot differentiate $(dependent_variable) with respect to " *
                        "$(variable)."
                    )
                )
                direction
            end
            inner_term, differentiated_variables, dependent_variable, Tuple(directions)
        end

    dependent_variable_arguments = @ignore_derivatives collect(
        SymbolicUtils.arguments(inner_term)
    )
    coordinates = _dependent_variable_coordinates(ctx, dependent_variable_arguments)
    derivative_operator = PINNDerivativeArrayOperator(
        ctx.derivative, ctx.phi, ctx.theta_axes, directions,
        ctx.dict_depvars[dependent_variable], ctx.multioutput
    )
    return derivative_operator(coordinates, ctx.theta)
end

function _lower_array_dependent_variable(ctx::PINNSymbolicArrayContext, term, name)
    dependent_variable_arguments = @ignore_derivatives collect(SymbolicUtils.arguments(term))
    coordinates = _dependent_variable_coordinates(ctx, dependent_variable_arguments)
    phi_operator = PINNPhiArrayOperator(
        ctx.phi, ctx.theta_axes, ctx.dict_depvars[name], ctx.multioutput
    )
    return phi_operator(coordinates, ctx.theta)
end

function _integral_variables(domain_variables)
    if SymbolicUtils.iscall(domain_variables)
        operation = SymbolicUtils.operation(domain_variables)
        if nameof(operation) === :tuple || operation === tuple
            return collect(SymbolicUtils.arguments(domain_variables))
        end
    elseif domain_variables isa Tuple || domain_variables isa AbstractVector
        return collect(domain_variables)
    end
    return [domain_variables]
end

function _integral_limit_metadata(limit, coordinate_indices, negative_infinity::Bool)
    symbol = try
        Symbolics.tosymbol(limit)
    catch
        nothing
    end
    if symbol isa Symbol && haskey(coordinate_indices, symbol)
        return 0.0, coordinate_indices[symbol], 0
    end

    value = _numeric_symbolic_value(limit)
    value === nothing && throw(
        PINNArrayLoweringError(
            "Integral limits must currently be numeric values or equation coordinates; " *
            "encountered $(limit)."
        )
    )
    infinity_kind = if isinf(value)
        negative_infinity && value == -Inf ? -1 : (!negative_infinity && value == Inf ? 1 : 0)
    else
        0
    end
    return Float64(value), 0, infinity_kind
end

function _lower_array_integral(ctx::PINNSymbolicArrayContext, term, operation)
    variable_indices, lower_values, upper_values, lower_coordinate_indices,
        upper_coordinate_indices, infinite_limits = @ignore_derivatives begin
        variables = _integral_variables(operation.domain.variables)
        variable_indices = map(variables) do variable
            symbol = Symbolics.tosymbol(variable)
            haskey(ctx.coordinate_indices, symbol) || throw(
                PINNArrayLoweringError(
                    "Integration variable $(symbol) is not present in the equation's " *
                    "coordinate array."
                )
            )
            ctx.coordinate_indices[symbol]
        end

        lower_limits, upper_limits = get_limits(operation.domain.domain)
        lower_values = Float64[]
        upper_values = Float64[]
        lower_coordinate_indices = Int[]
        upper_coordinate_indices = Int[]
        infinite_limits = Int[]
        for (lower, upper) in zip(lower_limits, upper_limits)
            lower_value, lower_coordinate, lower_infinity =
                _integral_limit_metadata(lower, ctx.coordinate_indices, true)
            upper_value, upper_coordinate, upper_infinity =
                _integral_limit_metadata(upper, ctx.coordinate_indices, false)
            push!(lower_values, lower_value)
            push!(upper_values, upper_value)
            push!(lower_coordinate_indices, lower_coordinate)
            push!(upper_coordinate_indices, upper_coordinate)
            push!(
                infinite_limits,
                lower_infinity == -1 && upper_infinity == 1 ? 2 :
                    (upper_infinity == 1 ? 1 : 0)
            )
        end
        Tuple(variable_indices), lower_values, upper_values, lower_coordinate_indices,
            upper_coordinate_indices, infinite_limits
    end

    integrand_expression = @ignore_derivatives first(SymbolicUtils.arguments(term))
    integrand = PINNRuntimeIntegrand(
        integrand_expression, ctx.theta_axes, ctx.phi, ctx.derivative,
        ctx.dict_indvars, ctx.dict_depvars, ctx.dict_depvar_input,
        ctx.coordinate_indices, ctx.parameter_indices, ctx.fixed_parameter_values,
        ctx.estimated_parameter_indices, ctx.multioutput
    )
    integral_operator = PINNIntegralArrayOperator(
        integrand, ctx.phi, ctx.derivative, ctx.theta_axes, variable_indices,
        lower_values, upper_values, lower_coordinate_indices,
        upper_coordinate_indices, infinite_limits
    )
    return integral_operator(ctx.cord, ctx.theta)
end

function _rebuild_array_expression(ctx::PINNSymbolicArrayContext, term)
    is_call = @ignore_derivatives SymbolicUtils.iscall(term)
    if !is_call
        expression = @ignore_derivatives toexpr(term)
        if expression isa Symbol && haskey(ctx.coordinate_indices, expression)
            row = ctx.coordinate_indices[expression]
            return PINNCoordinateSliceOperator(row)(ctx.cord, ctx.theta)
        elseif expression isa Symbol && haskey(ctx.estimated_parameter_indices, expression)
            return ctx.theta[ctx.estimated_parameter_indices[expression]]
        elseif expression isa Symbol && haskey(ctx.parameter_indices, expression)
            return ctx.fixed_parameters[ctx.parameter_indices[expression]]
        end
        numeric_value = @ignore_derivatives _numeric_symbolic_value(term)
        numeric_value === nothing || return numeric_value
        return term
    end

    operation = @ignore_derivatives SymbolicUtils.operation(term)
    operation isa Differential && return _lower_array_differential(ctx, term)
    operation isa Symbolics.Integral && return _lower_array_integral(ctx, term, operation)

    operation_expression = @ignore_derivatives toexpr(operation)
    if operation_expression isa Symbol && haskey(ctx.dict_depvars, operation_expression)
        return _lower_array_dependent_variable(ctx, term, operation_expression)
    end

    term_arguments = @ignore_derivatives collect(SymbolicUtils.arguments(term))
    lowered_arguments = map(
        argument -> _rebuild_array_expression(ctx, argument),
        term_arguments
    )
    if any(_is_array_expression, lowered_arguments)
        return broadcast(operation, lowered_arguments...)
    end

    if all(argument -> argument isa Number, lowered_arguments)
        return operation(lowered_arguments...)
    end

    return Num(
        SymbolicUtils.term(
            operation, Symbolics.unwrap.(lowered_arguments)...;
            type = @ignore_derivatives(SymbolicUtils.symtype(term))
        )
    )
end

function build_symbolic_array_residual(
        pinnrep::PINNRepresentation, equation, cord, theta, theta_axes,
        fixed_parameters, fixed_parameter_values, parameter_indices,
        estimated_parameter_indices;
        local_indvars = nothing
    )
    coordinate_indices = local_coordinate_index_map(
        equation, pinnrep.dict_indvars, pinnrep.dict_depvars, pinnrep.strategy,
        local_indvars, pinnrep.bcs
    )
    context = PINNSymbolicArrayContext(
        cord, theta, theta_axes, pinnrep.phi, pinnrep.derivative,
        pinnrep.dict_indvars, pinnrep.dict_depvars, pinnrep.dict_depvar_input,
        coordinate_indices, parameter_indices, fixed_parameters, fixed_parameter_values,
        estimated_parameter_indices, pinnrep.multioutput
    )
    normalized = normalize_equation_residual(equation, pinnrep.dict_depvars)
    residual = _rebuild_array_expression(context, Symbolics.unwrap(normalized))
    _is_symbolic_array(residual) || throw(
        PINNArrayLoweringError(
            "Equation $(equation) did not lower to a symbolic residual array."
        )
    )
    return residual
end

"""
    build_runtime_array_residual_function(pinnrep, equation; local_indvars = nothing)

Create the compatibility `(coordinates, theta) -> residual` interface from the native
array parser. This keeps public loss-inspection APIs available without invoking the legacy
per-equation `build_function`/`RuntimeGeneratedFunction` pipeline.
"""
function build_runtime_array_residual_function(
        pinnrep::PINNRepresentation, equation; local_indvars = nothing
    )
    theta_axes = getaxes(pinnrep.flat_init_params)
    clean_parameters = pinnrep.eq_params isa SciMLBase.NullParameters ?
        Any[] : collect(pinnrep.eq_params)
    parameter_indices = Dict{Symbol, Int}(
        Symbolics.tosymbol(parameter) => i
            for (i, parameter) in enumerate(clean_parameters)
    )
    estimated_parameter_indices = _estimated_parameter_indices(pinnrep, theta_axes)
    coordinate_indices = local_coordinate_index_map(
        equation, pinnrep.dict_indvars, pinnrep.dict_depvars, pinnrep.strategy,
        local_indvars, pinnrep.bcs
    )
    normalized = Symbolics.unwrap(
        normalize_equation_residual(equation, pinnrep.dict_depvars)
    )
    fixed_parameter_values = pinnrep.default_p === nothing ? Number[] : pinnrep.default_p

    return function (coordinates, theta)
        context = PINNSymbolicArrayContext(
            coordinates, theta, theta_axes, pinnrep.phi, pinnrep.derivative,
            pinnrep.dict_indvars, pinnrep.dict_depvars, pinnrep.dict_depvar_input,
            coordinate_indices, parameter_indices, fixed_parameter_values,
            fixed_parameter_values, estimated_parameter_indices, pinnrep.multioutput
        )
        return _rebuild_array_expression(context, normalized)
    end
end

function _symbolic_mse(residual)
    scalar_residual = Symbolics.scalarize(abs2.(residual))
    return sum(scalar_residual) / length(scalar_residual)
end

function _estimated_parameter_indices(pinnrep, theta_axes)
    pinnrep.param_estim || return Dict{Symbol, Int}()
    indexed_theta = ComponentArray(collect(eachindex(pinnrep.flat_init_params)), theta_axes)
    clean_parameters = collect(pinnrep.eq_params)
    return Dict(
        Symbolics.tosymbol(parameter) => indexed_theta.p[i]
            for (i, parameter) in enumerate(clean_parameters)
    )
end

"""
    build_pinn_optimization_system(pinnrep::PINNRepresentation)

Lower a PINN representation to symbolic residual arrays, a symbolic scalar
MSE objective, and a native `ModelingToolkit.OptimizationSystem`. The returned metadata
contains the operating-point map used to construct an `OptimizationProblem`.
"""
function build_pinn_optimization_system(pinnrep::PINNRepresentation)
    isempty(pinnrep.flat_init_params) && throw(
        PINNArrayLoweringError(
            "A native OptimizationSystem requires at least one optimization unknown."
        )
    )

    element_type = recursive_eltype(pinnrep.flat_init_params)
    training_sets, coordinate_provider = _optimization_training_sets(
        pinnrep, pinnrep.strategy, element_type
    )
    pde_training_sets, bc_training_sets = training_sets

    theta_axes = getaxes(pinnrep.flat_init_params)
    theta = _symbolic_unknown_vector(:pinn_theta, length(pinnrep.flat_init_params))

    clean_parameters = pinnrep.eq_params isa SciMLBase.NullParameters ?
        Any[] : collect(pinnrep.eq_params)
    parameter_indices = Dict{Symbol, Int}(
        Symbolics.tosymbol(parameter) => i
            for (i, parameter) in enumerate(clean_parameters)
    )
    estimated_parameter_indices = _estimated_parameter_indices(pinnrep, theta_axes)
    fixed_parameters = if pinnrep.param_estim || isempty(clean_parameters)
        nothing
    else
        _symbolic_parameter_vector(:pinn_fixed_parameters, length(clean_parameters))
    end

    pde_coordinates = map(enumerate(pde_training_sets)) do (i, values)
        _symbolic_parameter_matrix(Symbol("pinn_pde_coordinates_", i), size(values)...)
    end
    bc_coordinates = map(enumerate(bc_training_sets)) do (i, values)
        _symbolic_parameter_matrix(Symbol("pinn_bc_coordinates_", i), size(values)...)
    end

    pde_residuals = [
        build_symbolic_array_residual(
            pinnrep, equation, coordinate, theta, theta_axes, fixed_parameters,
            pinnrep.default_p, parameter_indices, estimated_parameter_indices;
            local_indvars
        )
            for (equation, coordinate, local_indvars) in
            zip(pinnrep.eqs, pde_coordinates, pinnrep.pde_indvars)
    ]
    bc_residuals = [
        build_symbolic_array_residual(
            pinnrep, equation, coordinate, theta, theta_axes, fixed_parameters,
            pinnrep.default_p, parameter_indices, estimated_parameter_indices;
            local_indvars
        )
            for (equation, coordinate, local_indvars) in
            zip(pinnrep.bcs, bc_coordinates, pinnrep.bc_indvars)
    ]

    pde_losses = _symbolic_mse.(pde_residuals)
    bc_losses = _symbolic_mse.(bc_residuals)

    pde_weight_parameters = isempty(pde_losses) ? nothing :
        _symbolic_parameter_vector(:pinn_pde_loss_weights, length(pde_losses))
    bc_weight_parameters = isempty(bc_losses) ? nothing :
        _symbolic_parameter_vector(:pinn_bc_loss_weights, length(bc_losses))
    additional_weight_parameters = pinnrep.additional_loss === nothing ? nothing :
        _symbolic_parameter_vector(:pinn_additional_loss_weight, 1)

    weighted_losses = Any[
        pde_weight_parameters[i] * loss for (i, loss) in enumerate(pde_losses)
    ]
    append!(weighted_losses, Any[
        bc_weight_parameters[i] * loss for (i, loss) in enumerate(bc_losses)
    ])
    if pinnrep.additional_loss !== nothing
        additional_operator = PINNAdditionalLossOperator(
            pinnrep.additional_loss, pinnrep.phi, theta_axes, pinnrep.param_estim
        )
        push!(
            weighted_losses,
            additional_weight_parameters[1] * additional_operator(theta)
        )
    end
    symbolic_cost = sum(weighted_losses)

    system_parameters = Any[]
    append!(system_parameters, pde_coordinates)
    append!(system_parameters, bc_coordinates)
    fixed_parameters === nothing || push!(system_parameters, fixed_parameters)
    pde_weight_parameters === nothing || push!(system_parameters, pde_weight_parameters)
    bc_weight_parameters === nothing || push!(system_parameters, bc_weight_parameters)
    additional_weight_parameters === nothing ||
        push!(system_parameters, additional_weight_parameters)
    source_system = ModelingToolkit.OptimizationSystem(
        symbolic_cost, [theta], system_parameters; name = :pinn_optimization_system
    )
    compiled_system = mtkcompile(source_system)

    operating_point = Pair{Any, Any}[
        theta => _mtk_operating_point_value(pinnrep.flat_init_params)
    ]
    append!(
        operating_point,
        Pair.(pde_coordinates, _mtk_operating_point_value.(pde_training_sets))
    )
    append!(
        operating_point,
        Pair.(bc_coordinates, _mtk_operating_point_value.(bc_training_sets))
    )
    if fixed_parameters !== nothing
        pinnrep.default_p === nothing && throw(
            PINNArrayLoweringError(
                "Fixed PDE parameters require numeric defaults in the PDESystem."
            )
        )
        push!(
            operating_point,
            fixed_parameters => _mtk_operating_point_value(pinnrep.default_p)
        )
    end
    pde_weight_parameters === nothing || push!(
        operating_point,
        pde_weight_parameters => _mtk_operating_point_value(pinnrep.adaloss.pde_loss_weights)
    )
    bc_weight_parameters === nothing || push!(
        operating_point,
        bc_weight_parameters => _mtk_operating_point_value(pinnrep.adaloss.bc_loss_weights)
    )
    additional_weight_parameters === nothing || push!(
        operating_point,
        additional_weight_parameters =>
            _mtk_operating_point_value(pinnrep.adaloss.additional_loss_weights)
    )

    runs_on_cpu = safe_get_device(pinnrep.flat_init_params) == cdev
    executable = !(pinnrep.strategy isa QuadratureTraining) && runs_on_cpu
    fallback_reason = executable ? nothing :
        if pinnrep.strategy isa QuadratureTraining
            "The symbolic system is available, but the existing strategy objective is " *
            "retained for quadrature execution."
        else
            "The symbolic system is available, but the existing strategy objective is " *
            "retained for GPU execution because MTK scalar codegen indexes CuArray " *
            "unknown vectors on the host."
        end

    data = PINNOptimizationSystemData(
        source_system, compiled_system, theta,
        (; pde = pde_coordinates, bc = bc_coordinates), fixed_parameters,
        (;
            pde = pde_weight_parameters, bc = bc_weight_parameters,
            additional = additional_weight_parameters,
        ), operating_point, coordinate_provider, executable, fallback_reason
    )
    return (;
        pde_residuals, bc_residuals, pde_losses, bc_losses, weighted_losses,
        cost = symbolic_cost, system = source_system, compiled_system, data
    )
end
