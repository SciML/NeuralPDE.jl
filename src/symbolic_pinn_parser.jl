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

using ChainRulesCore: ChainRulesCore, NoTangent

function _replace_index(x::AbstractVector, i::Integer, val)
    return [j == i ? val : x[j] for j in eachindex(x)]
end

function ChainRulesCore.rrule(::typeof(_replace_index), x::AbstractVector, i::Integer, val)
    y = _replace_index(x, i, val)
    function _replace_index_pullback(Δ)
        dx = [j == i ? zero(Δ[j]) : Δ[j] for j in eachindex(x)]
        dval = Δ[i]
        return NoTangent(), dx, NoTangent(), dval
    end
    return y, _replace_index_pullback
end

function _perturbed_vector(x::AbstractVector, dir::Integer, h)
    return [j == dir ? x[j] + h : x[j] for j in eachindex(x)]
end

function ChainRulesCore.rrule(::typeof(_perturbed_vector), x::AbstractVector, dir::Integer, h)
    y = _perturbed_vector(x, dir, h)
    function _perturbed_vector_pullback(Δ)
        return NoTangent(), Δ, NoTangent(), Δ[dir]
    end
    return y, _perturbed_vector_pullback
end

function _perturbed_vector(x::AbstractVector, dir1::Integer, h1, dir2::Integer, h2)
    if dir1 == dir2
        return _perturbed_vector(x, dir1, h1 + h2)
    else
        return [j == dir1 ? x[j] + h1 : (j == dir2 ? x[j] + h2 : x[j]) for j in eachindex(x)]
    end
end

function ChainRulesCore.rrule(::typeof(_perturbed_vector), x::AbstractVector, dir1::Integer, h1, dir2::Integer, h2)
    y = _perturbed_vector(x, dir1, h1, dir2, h2)
    function _perturbed_vector_pullback(Δ)
        return NoTangent(), Δ, NoTangent(), Δ[dir1], NoTangent(), Δ[dir2]
    end
    return y, _perturbed_vector_pullback
end

function _multi_direction_derivative(f, x::AbstractVector, directions::AbstractVector{<:Integer})
    ndirs = length(directions)
    T = eltype(x)
    h = T(1e-4)

    if ndirs == 0
        return f(x)
    elseif ndirs == 1
        dir = directions[1]
        x_p = _perturbed_vector(x, dir, h)
        x_m = _perturbed_vector(x, dir, -h)
        return (f(x_p) - f(x_m)) / (2h)
    elseif ndirs == 2
        dir1, dir2 = directions[1], directions[2]
        if dir1 == dir2
            x_p = _perturbed_vector(x, dir1, h)
            x_m = _perturbed_vector(x, dir1, -h)
            return (f(x_p) - 2 * f(x) + f(x_m)) / (h^2)
        else
            x_pp = _perturbed_vector(x, dir1, h, dir2, h)
            x_pm = _perturbed_vector(x, dir1, h, dir2, -h)
            x_mp = _perturbed_vector(x, dir1, -h, dir2, h)
            x_mm = _perturbed_vector(x, dir1, -h, dir2, -h)
            return (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4h^2)
        end
    else
        # Fallback for 3rd order or higher
        dir = first(directions)
        g = a -> _multi_direction_derivative(
            f, _replace_index(x, dir, a), @view(directions[2:end])
        )
        return (g(x[dir] + h) - g(x[dir] - h)) / (2h)
    end
end



function (f::SymbolicPINNDerivativeWrapper)(input::AbstractVector, p, directions)
    x = collect(input)
    dirs = Int.(directions)
    val = _multi_direction_derivative(z -> first(f.nn(z, p)), x, dirs)
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

function _symbolic_pinn_neural_specs(chains, n_input, n_dvs; init_params = nothing)
    chain_vec = _chain_vector(chains)
    length(chain_vec) == n_dvs ||
        throw(ArgumentError("Expected one neural network chain per dependent variable."))

    return map(enumerate(chain_vec)) do (i, ch)
        nn_name = n_dvs == 1 ? :NN : Symbol(:NN_, i)
        p_name = n_dvs == 1 ? :p : Symbol(:p_, i)
        dnn_name = n_dvs == 1 ? :DNN : Symbol(:DNN_, i)
        snn_kwargs = (;
            chain = ch, n_input = n_input, n_output = 1,
            nn_name = nn_name, nn_p_name = p_name
        )
        if init_params !== nothing
            # init_params can be a vector (one per DV) or a single ComponentArray
            p_init = init_params isa AbstractVector{<:AbstractArray} ? init_params[i] : init_params
            snn_kwargs = (; snn_kwargs..., init_params = p_init)
        end
        nn, p = SymbolicNeuralNetwork(; snn_kwargs...)
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

function _derivative_directions(derivative_vars, ivs)
    iv_terms = Symbolics.unwrap.(ivs)
    return map(derivative_vars) do var
        idx = findfirst(iv -> isequal(iv, var), iv_terms)
        idx === nothing &&
            throw(ArgumentError("Derivative variable $var is not an independent variable."))
        idx
    end
end

"""
    _prewalk_substitute(expr, dv_ops, ivs, neural_specs)

Single-pass prewalk substitution of dependent-variable calls in a symbolic expression.

Uses `SymbolicUtils.Rewriters.Prewalk` to traverse the expression tree top-down (preorder).
At each node, a nested matcher function checks:
- If the node is a chain of `Differential` applications wrapping a DV call, it is
  immediately replaced by the symbolic derivative neural-network call.
- If the node is a bare DV call (no Differential wrapper), it is immediately replaced
  by the symbolic value neural-network call.
- Otherwise, the node is returned unchanged, and `Prewalk` recursively traverses its children.
"""
function _prewalk_substitute(expr, dv_ops, ivs, neural_specs)
    matcher = function (node)
        # --- Prewalk priority 1: Differential-wrapped DV chain ---
        deriv_info = _as_dv_derivative(node, dv_ops)
        if deriv_info !== nothing
            spec = neural_specs[deriv_info.dv_index]
            args = collect(SymbolicUtils.arguments(deriv_info.call))
            directions = _derivative_directions(deriv_info.derivative_vars, ivs)
            replacement = spec.derivative(args, spec.parameters, directions)[1]
            return Symbolics.unwrap(replacement)
        end

        # --- Prewalk priority 2: Bare DV call (no Differential wrapper) ---
        dv_index = _matching_dv_index(node, dv_ops)
        if dv_index !== nothing
            spec = neural_specs[dv_index]
            args = collect(SymbolicUtils.arguments(node))
            replacement = spec.value(args, spec.parameters)[1]
            return Symbolics.unwrap(replacement)
        end

        return node
    end

    inbuilt_rewriter = SymbolicUtils.Rewriters.Prewalk(matcher)
    return inbuilt_rewriter(expr)
end


"""
    symbolic_pinn_residual(eq, ivs, dvs, neural_specs)

Create a symbolic PINN residual for one ModelingToolkit equation by replacing dependent
variable calls and derivative calls with symbolic neural-network calls using a
single-pass prewalk substitution.
"""
function symbolic_pinn_residual(eq, ivs, dvs, neural_specs)
    raw = _equation_residual(eq)
    expr = Symbolics.unwrap(raw)
    dv_ops = _dv_operation.(dvs)
    substituted = _prewalk_substitute(expr, dv_ops, ivs, neural_specs)
    return Num(substituted)
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

function _theta0(specs::AbstractVector{<:SymbolicPINNNeuralSpec})
    return vcat([_theta0(spec) for spec in specs]...)
end

function _split_theta(theta, param_lengths)
    offsets = cumsum(param_lengths)
    return ntuple(length(param_lengths)) do i
        lo = i == 1 ? 1 : offsets[i - 1] + 1
        @view(theta[lo:offsets[i]])
    end
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
    param_lengths = [length(Symbolics.getdefaultval(spec.parameters)) for spec in neural_specs]

    return function (point, theta)
        param_views = _split_theta(theta, param_lengths)
        return compiled(point..., runtime_args..., param_views...)
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

Build a symbolic PINN loss for a `PDESystem`. Supports single and multiple dependent
variables, same-direction and mixed/cross-direction derivatives. The returned object is
a named tuple containing symbolic residuals, lowered residual functions, sampled points,
initial parameters, and simple mean-squared PDE/BC/full loss functions.
"""
function build_symbolic_pinn_loss(sys::PDESystem, chain; n_interior::Integer = 64,
        n_bc::Integer = 64)
    parsed = parse_pde_system(sys)

    neural_specs = _symbolic_pinn_neural_specs(chain, length(parsed.ivs), length(parsed.dvs))
    theta0 = _theta0(neural_specs)

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
