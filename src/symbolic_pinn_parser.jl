struct SymbolicPINNSystem
    sys
    eqs
    bcs
    domains
    ivs
    dvs
    ps
end

struct SymbolicPINNDerivativeWrapper{F}
    nn::F
end

struct SymbolicPINNNeuralSpec
    value
    derivative
    parameters
end

"""
    parse_pde_system(sys::PDESystem)

Collect the ModelingToolkit `PDESystem` pieces needed by the experimental symbolic PINN
parser. This intentionally uses ModelingToolkit accessors instead of direct field access.
"""
function parse_pde_system(sys::PDESystem)
    return SymbolicPINNSystem(
        sys,
        ModelingToolkit.get_eqs(sys),
        ModelingToolkit.get_bcs(sys),
        ModelingToolkit.get_domain(sys),
        ModelingToolkit.get_ivs(sys),
        ModelingToolkit.get_dvs(sys),
        ModelingToolkit.get_ps(sys)
    )
end

function _replace_index(x::AbstractVector, i::Integer, val)
    x_ = Vector{promote_type(eltype(x), typeof(val))}(x)
    x_[i] = val
    return x_
end

function _same_direction_derivative(f, x::AbstractVector, direction::Integer, order::Integer)
    order == 0 && return f(x)
    return ForwardDiff.derivative(
        a -> _same_direction_derivative(
            f, _replace_index(x, direction, a), direction, order - 1
        ),
        x[direction]
    )
end

function (f::SymbolicPINNDerivativeWrapper)(input::AbstractVector, p, direction, order)
    x = collect(input)
    dir = Int(direction)
    ord = Int(order)
    val = _same_direction_derivative(z -> first(f.nn(z, p)), x, dir, ord)
    return [val]
end

function _symbolic_pinn_derivative_operator(nn; name = :DNN)
    wrapper = SymbolicPINNDerivativeWrapper(Symbolics.getdefaultval(nn))
    DNN = @parameters ($(name)::typeof(wrapper))(..)[1:1] = wrapper [tunable = false]
    return only(DNN)
end

function _chain_vector(chain)
    return chain isa AbstractVector ? collect(chain) : [chain]
end

function _symbolic_pinn_neural_specs(chains, n_input, n_dvs)
    chain_vec = _chain_vector(chains)
    length(chain_vec) == n_dvs ||
        throw(ArgumentError("Expected one neural network chain per dependent variable."))

    return map(enumerate(chain_vec)) do (i, ch)
        nn_name = n_dvs == 1 ? :NN : Symbol(:NN_, i)
        p_name = n_dvs == 1 ? :p : Symbol(:p_, i)
        dnn_name = n_dvs == 1 ? :DNN : Symbol(:DNN_, i)
        nn, p = SymbolicNeuralNetwork(;
            chain = ch, n_input = n_input, n_output = 1,
            nn_name = nn_name, nn_p_name = p_name
        )
        SymbolicPINNNeuralSpec(nn, _symbolic_pinn_derivative_operator(nn; name = dnn_name), p)
    end
end

function _equation_residual(eq)
    return Symbolics.expand_derivatives(eq.lhs - eq.rhs)
end

function _dv_operation(dv)
    unwrapped = Symbolics.unwrap(dv)
    return SymbolicUtils.iscall(unwrapped) ? SymbolicUtils.operation(unwrapped) : unwrapped
end

function _matching_dv_index(expr, dv_ops)
    SymbolicUtils.iscall(expr) || return nothing
    op = SymbolicUtils.operation(expr)
    return findfirst(dv_op -> isequal(op, dv_op), dv_ops)
end

function _as_dv_derivative(expr, dv_ops)
    SymbolicUtils.iscall(expr) || return nothing
    current = expr
    derivative_vars = Any[]

    while SymbolicUtils.iscall(current) &&
            SymbolicUtils.operation(current) isa Differential
        D = SymbolicUtils.operation(current)
        append!(derivative_vars, fill(D.x, Int(D.order)))
        args = SymbolicUtils.arguments(current)
        length(args) == 1 || return nothing
        current = only(args)
    end

    isempty(derivative_vars) && return nothing
    dv_index = _matching_dv_index(current, dv_ops)
    dv_index === nothing && return nothing
    return (dv_index = dv_index, term = expr, call = current, derivative_vars = derivative_vars)
end

function _find_dv_calls!(calls, expr, dv_ops)
    SymbolicUtils.iscall(expr) || return calls
    _as_dv_derivative(expr, dv_ops) !== nothing && return calls

    dv_index = _matching_dv_index(expr, dv_ops)
    if dv_index !== nothing
        push!(calls, (dv_index, expr))
        return calls
    end

    for arg in SymbolicUtils.arguments(expr)
        _find_dv_calls!(calls, arg, dv_ops)
    end
    return calls
end

function _find_derivative_dv_calls!(calls, expr, dv_ops)
    SymbolicUtils.iscall(expr) || return calls

    derivative_call = _as_dv_derivative(expr, dv_ops)
    if derivative_call !== nothing
        push!(calls, derivative_call)
        return calls
    end

    for arg in SymbolicUtils.arguments(expr)
        _find_derivative_dv_calls!(calls, arg, dv_ops)
    end
    return calls
end

function _derivative_direction_and_order(derivative_vars, ivs)
    iv_terms = Symbolics.unwrap.(ivs)
    directions = map(derivative_vars) do var
        idx = findfirst(iv -> isequal(iv, var), iv_terms)
        idx === nothing &&
            throw(ArgumentError("Derivative variable $var is not an independent variable."))
        idx
    end

    first_direction = first(directions)
    all(==(first_direction), directions) ||
        throw(ArgumentError("The symbolic PINN MVP currently supports same-direction derivatives only."))
    return first_direction, length(directions)
end

function _substitute_residual(raw, substitutions)
    return Symbolics.substitute_in_deriv_and_depvar(raw, substitutions)
end

"""
    symbolic_pinn_residual(eq, ivs, dvs, neural_specs)

Create a symbolic PINN residual for one ModelingToolkit equation by replacing dependent
variable calls and same-direction derivative calls with symbolic neural-network calls.
"""
function symbolic_pinn_residual(eq, ivs, dvs, neural_specs)
    raw = _equation_residual(eq)
    expr = Symbolics.unwrap(raw)
    dv_ops = _dv_operation.(dvs)

    derivative_calls = Any[]
    _find_derivative_dv_calls!(derivative_calls, expr, dv_ops)

    value_calls = Tuple{Int, Any}[]
    _find_dv_calls!(value_calls, expr, dv_ops)

    substitutions = Dict{Any, Any}()
    for call in derivative_calls
        spec = neural_specs[call.dv_index]
        args = collect(SymbolicUtils.arguments(call.call))
        direction, order = _derivative_direction_and_order(call.derivative_vars, ivs)
        substitutions[call.term] = spec.derivative(args, spec.parameters, direction, order)[1]
    end

    for (dv_index, call) in value_calls
        spec = neural_specs[dv_index]
        args = collect(SymbolicUtils.arguments(call))
        substitutions[call] = spec.value(args, spec.parameters)[1]
    end

    return _substitute_residual(raw, substitutions)
end

function _contains_dv_call(expr, dvs)
    dv_ops = _dv_operation.(dvs)
    found = Ref(false)

    function walk(ex)
        SymbolicUtils.iscall(ex) || return
        if _matching_dv_index(ex, dv_ops) !== nothing
            found[] = true
            return
        end
        for arg in SymbolicUtils.arguments(ex)
            walk(arg)
            found[] && return
        end
    end

    walk(Symbolics.unwrap(expr))
    return found[]
end

function _theta0(spec::SymbolicPINNNeuralSpec)
    return Vector(Symbolics.getdefaultval(spec.parameters))
end

function _runtime_args(neural_specs)
    nn_defaults = map(spec -> Symbolics.getdefaultval(spec.value), neural_specs)
    dnn_defaults = map(spec -> Symbolics.getdefaultval(spec.derivative), neural_specs)
    return (nn_defaults..., dnn_defaults...)
end

function _compiled_residual(residual, ivs, neural_specs)
    iv_args = Symbolics.unwrap.(ivs)
    nn_args = map(spec -> spec.value, neural_specs)
    dnn_args = map(spec -> spec.derivative, neural_specs)
    p_args = map(spec -> spec.parameters, neural_specs)
    compiled = Symbolics.build_function(
        residual, iv_args..., nn_args..., dnn_args..., p_args...;
        expression = Val(false)
    )
    runtime_args = _runtime_args(neural_specs)

    return function (point, theta)
        return compiled(point..., runtime_args..., theta)
    end
end

function _domain_bounds(domains)
    return map(domains) do domain
        (infimum(domain.domain), supremum(domain.domain))
    end
end

function _axis_points(lo, hi, n::Integer; interior::Bool)
    n > 0 || throw(ArgumentError("Number of collocation points must be positive."))
    if interior
        return collect(range(lo, hi; length = n + 2))[2:(end - 1)]
    else
        return n == 1 ? [(lo + hi) / 2] : collect(range(lo, hi; length = n))
    end
end

function _collocation_points(domains, n::Integer; interior::Bool)
    axes = [_axis_points(lo, hi, n; interior) for (lo, hi) in _domain_bounds(domains)]
    grid = vec([collect(Float64, point) for point in Iterators.product(axes...)])
    return reduce(hcat, grid)  # (D, N) matrix
end

_scalar_residual_value(x::Number) = x
_scalar_residual_value(x) = first(x)

"""
    _wrap_as_datafree(compiled_residual_fn)

Wrap a compiled scalar residual function into the `(cord::Matrix, θ) -> Matrix` format
expected by NeuralPDE's training strategies (`GridTraining`, `StochasticTraining`, etc.).

The returned function accepts a `(D, N)` coordinate matrix and a parameter vector,
and returns a `(1, N)` matrix of residual values, compatible with
`merge_strategy_with_loss_function` and `get_loss_function`.
"""
function _wrap_as_datafree(compiled_residual_fn)
    return function (cord, theta)
        N = size(cord, 2)
        out = map(1:N) do j
            _scalar_residual_value(compiled_residual_fn(@view(cord[:, j]), theta))
        end
        return reshape(out, 1, :)
    end
end

function _mean_square(datafree_fns, points_matrix, theta)
    isempty(datafree_fns) && return zero(eltype(theta))
    return sum(datafree_fns) do f
        mean(abs2, f(points_matrix, theta))
    end / length(datafree_fns)
end

"""
    build_symbolic_pinn_loss(sys::PDESystem, chain; n_interior = 64, n_bc = 64)

Build an experimental symbolic PINN loss for the heat-equation MVP. The returned object is
a named tuple containing symbolic residuals, lowered residual functions, sampled points,
initial parameters, and simple mean-squared PDE/BC/full loss functions.
"""
function build_symbolic_pinn_loss(sys::PDESystem, chain; n_interior::Integer = 64,
        n_bc::Integer = 64)
    parsed = parse_pde_system(sys)
    length(parsed.dvs) == 1 ||
        throw(ArgumentError("The symbolic PINN MVP currently supports one dependent variable."))

    neural_specs = _symbolic_pinn_neural_specs(chain, length(parsed.ivs), length(parsed.dvs))
    theta0 = _theta0(only(neural_specs))

    pde_residuals = [
        symbolic_pinn_residual(eq, parsed.ivs, parsed.dvs, neural_specs)
            for eq in parsed.eqs
    ]
    bc_residuals = [
        symbolic_pinn_residual(bc, parsed.ivs, parsed.dvs, neural_specs)
            for bc in parsed.bcs
    ]

    pde_functions = [_compiled_residual(res, parsed.ivs, neural_specs) for res in pde_residuals]
    bc_functions = [_compiled_residual(res, parsed.ivs, neural_specs) for res in bc_residuals]

    datafree_pde_loss_functions = [_wrap_as_datafree(f) for f in pde_functions]
    datafree_bc_loss_functions = [_wrap_as_datafree(f) for f in bc_functions]

    pde_points = _collocation_points(parsed.domains, n_interior; interior = true)
    bc_points = _collocation_points(parsed.domains, n_bc; interior = false)

    pde_loss = theta -> _mean_square(datafree_pde_loss_functions, pde_points, theta)
    bc_loss = theta -> _mean_square(datafree_bc_loss_functions, bc_points, theta)
    loss = theta -> pde_loss(theta) + bc_loss(theta)

    return (
        parsed = parsed,
        neural_specs = neural_specs,
        theta0 = theta0,
        pde_residuals = pde_residuals,
        bc_residuals = bc_residuals,
        residual_functions = (pde = pde_functions, bc = bc_functions),
        datafree_pde_loss_functions = datafree_pde_loss_functions,
        datafree_bc_loss_functions = datafree_bc_loss_functions,
        points = (pde = pde_points, bc = bc_points),
        pde_loss = pde_loss,
        bc_loss = bc_loss,
        loss = loss
    )
end
