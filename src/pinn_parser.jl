struct SymbolicPINNSystem
    sys::ModelingToolkit.PDESystem
    eqs::Vector{Symbolics.Equation}
    bcs::Vector{Symbolics.Equation}
    domains::Vector{Symbolics.VarDomainPairing}
    ivs::Vector{Symbolics.Num}
    dvs::Vector{Symbolics.Num}
    ps::Vector{Symbolics.Num}
end

struct SymbolicPINNNeuralSpec{V, P}
    value::V
    parameters::P
end

struct SymbolicPINNIntegrandInfo{S, T, L, U, I}
    integrand_substituted::S
    τs::T
    lb::L
    ub::U
    integrating_var_indices::I
end

"""
    parse_pde_system(sys::PDESystem)

Collect the ModelingToolkit `PDESystem` pieces needed by the experimental symbolic PINN
parser. This intentionally uses ModelingToolkit accessors instead of direct field access.
"""
function parse_pde_system(sys::PDESystem)
    raw_ps = ModelingToolkit.get_ps(sys)
    ps_vec = raw_ps isa SciMLBase.NullParameters ? Symbolics.Num[] : collect(Symbolics.Num, raw_ps)
    return SymbolicPINNSystem(
        sys,
        ModelingToolkit.get_eqs(sys),
        ModelingToolkit.get_bcs(sys),
        ModelingToolkit.get_domain(sys),
        ModelingToolkit.get_ivs(sys),
        ModelingToolkit.get_dvs(sys),
        ps_vec
    )
end

using ChainRulesCore: ChainRulesCore, NoTangent
using Symbolics: Symbolics, Differential, Integral
import DomainSets
using Integrals: Integrals, CubatureJLh
# No BatchVector needed — the pure BasicSymbolic pipeline avoids
# batched NN wrapper gymnastics entirely.

function _get_limits(domain)
    if domain isa DomainSets.AbstractInterval
        return [DomainSets.leftendpoint(domain)], [DomainSets.rightendpoint(domain)]
    elseif domain isa DomainSets.ProductDomain
        return collect(map(DomainSets.leftendpoint, DomainSets.components(domain))),
            collect(map(DomainSets.rightendpoint, DomainSets.components(domain)))
    end
    throw(ArgumentError("Unsupported integration domain: $domain"))
end

function _integrating_variables(op_domain_variables, ivs)
    unwrapped_vars = Symbolics.unwrap(op_domain_variables)
    vars = if unwrapped_vars isa Tuple
        collect(unwrapped_vars)
    elseif unwrapped_vars isa AbstractVector
        collect(unwrapped_vars)
    elseif SymbolicUtils.iscall(unwrapped_vars) && SymbolicUtils.operation(unwrapped_vars) === tuple
        SymbolicUtils.arguments(unwrapped_vars)
    else
        [unwrapped_vars]
    end
    unwrapped_ivs = Symbolics.unwrap.(ivs)
    
    return map(vars) do v
        unwrapped_v = Symbolics.unwrap(v)
        idx = findfirst(iv -> isequal(iv, unwrapped_v), unwrapped_ivs)
        idx === nothing && throw(ArgumentError("Integrating variable $v (unwrapped: $unwrapped_v) is not an independent variable of the system."))
        idx
    end
end

function SymbolicPINNIntegralPlaceholder(args...)
    # Dummy placeholder function used in symbolic tree before compilation
end

function _get_value_at(x::AbstractMatrix, i)
    return size(x, 1) == 1 ? x[1, i] : x[:, i]
end
function _get_value_at(x::AbstractVector, i)
    return x[i]
end

function _get_value_at(x, i)
    return x
end

function _solve_pinn_integral(integrand_fn, num_bounds::Int, rest...)
    idx_num_ivs = 2 * num_bounds + 1
    num_ivs::Int = rest[idx_num_ivs]
    args_raw = rest[(idx_num_ivs + 1):end]
    args_tuple = Tuple(args_raw)

    is_batch = false
    N = 1
    for j in 1:num_ivs
        val = args_raw[j]
        if val isa AbstractMatrix
            is_batch = true
            N = size(val, 2)
            break
        elseif val isa AbstractVector
            is_batch = true
            N = length(val)
            break
        end
    end
    for j in 1:(2*num_bounds)
        val = rest[j]
        if val isa AbstractMatrix && size(val, 2) > 1
            is_batch = true
            N = size(val, 2)
            break
        elseif val isa AbstractVector && length(val) > 1
            is_batch = true
            N = length(val)
            break
        end
    end

    if is_batch
        results = map(1:N) do i
            point_i = ntuple(j -> j <= num_ivs ? _get_value_at(args_raw[j], i) : args_raw[j], length(args_raw))
            if num_bounds == 1
                lb_val = _get_value_at(rest[1], i)
                ub_val = _get_value_at(rest[2], i)
                lb_val = lb_val isa AbstractVector ? lb_val[1] : lb_val
                ub_val = ub_val isa AbstractVector ? ub_val[1] : ub_val
            else
                lb_val = [_get_value_at(rest[j], i) for j in 1:num_bounds]
                ub_val = [_get_value_at(rest[num_bounds + j], i) for j in 1:num_bounds]
            end
            has_inf = (lb_val isa Number ? (isinf(lb_val) || isinf(ub_val)) : (any(isinf, lb_val) || any(isinf, ub_val)))
            lb_clean = has_inf ? (lb_val isa Number ? (isinf(lb_val) ? (lb_val > 0 ? 100.0 : -100.0) : lb_val) : map(b -> isinf(b) ? (b > 0 ? 100.0 : -100.0) : b, lb_val)) : lb_val
            ub_clean = has_inf ? (ub_val isa Number ? (isinf(ub_val) ? (ub_val > 0 ? 100.0 : -100.0) : ub_val) : map(b -> isinf(b) ? (b > 0 ? 100.0 : -100.0) : b, ub_val)) : ub_val
            
            integrand = if num_bounds > 1
                (τ, p_) -> begin
                    val = first(integrand_fn(ntuple(k -> τ[k], Val(num_bounds))..., point_i...))
                    isnan(val) ? 0.0 : val
                end
            else
                (τ, p_) -> begin
                    val = first(integrand_fn(τ, point_i...))
                    isnan(val) ? 0.0 : val
                end
            end
            prob = Integrals.IntegralProblem(integrand, (lb_clean, ub_clean))
            alg = has_inf ? Integrals.QuadGKJL() : Integrals.CubatureJLh()
            sol = Integrals.solve(prob, alg, reltol = 1e-3, abstol = 1e-3)
            sol.u isa AbstractArray ? first(sol.u) : sol.u
        end
        return length(results) == 1 ? results[1] : results
    else
        if num_bounds == 1
            lb_val = rest[1] isa AbstractVector ? rest[1][1] : rest[1]
            ub_val = rest[2] isa AbstractVector ? rest[2][1] : rest[2]
        else
            lb_val = [rest[j] for j in 1:num_bounds]
            ub_val = [rest[num_bounds + j] for j in 1:num_bounds]
        end
        
        has_inf = (lb_val isa Number ? (isinf(lb_val) || isinf(ub_val)) : (any(isinf, lb_val) || any(isinf, ub_val)))
        lb_clean = has_inf ? (lb_val isa Number ? (isinf(lb_val) ? (lb_val > 0 ? 100.0 : -100.0) : lb_val) : map(b -> isinf(b) ? (b > 0 ? 100.0 : -100.0) : b, lb_val)) : lb_val
        ub_clean = has_inf ? (ub_val isa Number ? (isinf(ub_val) ? (ub_val > 0 ? 100.0 : -100.0) : ub_val) : map(b -> isinf(b) ? (b > 0 ? 100.0 : -100.0) : b, ub_val)) : ub_val
        
        args_scalar = ntuple(j -> j <= num_ivs ? _get_value_at(args_raw[j], 1) : args_raw[j], length(args_raw))
        integrand = if num_bounds > 1
            (τ, p_) -> begin
                val = first(integrand_fn(ntuple(k -> τ[k], Val(num_bounds))..., args_scalar...))
                isnan(val) ? 0.0 : val
            end
        else
            (τ, p_) -> begin
                val = first(integrand_fn(τ, args_scalar...))
                isnan(val) ? 0.0 : val
            end
        end
        prob = Integrals.IntegralProblem(integrand, (lb_clean, ub_clean))
        alg = has_inf ? Integrals.QuadGKJL() : Integrals.CubatureJLh()
        sol = Integrals.solve(prob, alg, reltol = 1e-3, abstol = 1e-3)
        u_val = sol.u isa AbstractArray ? first(sol.u) : sol.u
        return u_val
    end
end

# No SymbolicPINNValueWrapper needed — NN evaluation stays in the
# Symbolics domain as SymbolicNeuralNetwork callable terms.
# The actual Lux chain is bound at compile time via Symbolics.getdefaultval.

# ---------- Expr manipulation removed ----------
# _is_pinn_dottable, _dot_pinn, _extract_arg_names, _arg_name have been
# eliminated.  Broadcasting is now handled by per-point symbolic substitution
# in the BasicSymbolic domain — no manual Expr AST editing required.

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
        in_dim = n_input isa AbstractVector ? n_input[i] : n_input
        snn_kwargs = (;
            chain = ch, n_input = in_dim, n_output = 1,
            nn_name = nn_name, nn_p_name = p_name
        )
        if init_params !== nothing
            # init_params can be a vector (one per DV) or a single ComponentArray
            p_init = init_params isa AbstractVector{<:AbstractArray} ? init_params[i] : init_params
            snn_kwargs = (; snn_kwargs..., init_params = p_init)
        end
        nn, p = SymbolicNeuralNetwork(; snn_kwargs...)
        SymbolicPINNNeuralSpec(nn, p)
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
    for (i, dv_op) in enumerate(dv_ops)
        isequal(op, dv_op)::Bool && return i
    end
    return nothing
end

function _as_dv_derivative(expr, dv_ops)
    SymbolicUtils.iscall(expr) || return nothing
    current = expr
    derivative_vars = Symbolics.SymbolicT[]

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

function _symbolic_derivative_fd(spec, args, directions, ivs; ε = nothing)
    order = length(directions)
    step_size = if ε isa Nothing || ε === 1e-8
        eps(Float64) ^ (1 / (2 + order))
    else
        ε
    end
    
    # Use explicit stencils for same-direction derivatives up to order 4
    if !isempty(directions) && all(d -> d == directions[1], directions)
        dir = directions[1]
        shift(k) = [j == dir ? args[j] + k * step_size : args[j] for j in eachindex(args)]
        eval_at(k) = spec.value(shift(k), spec.parameters)[1]
        
        if order == 1
            return (eval_at(1) - eval_at(-1)) / (2 * step_size)
        elseif order == 2
            return (eval_at(1) + eval_at(-1) - 2 * eval_at(0)) / (step_size^2)
        elseif order == 3
            return (eval_at(2) - 2 * eval_at(1) + 2 * eval_at(-1) - eval_at(-2)) / (2 * step_size^3)
        elseif order == 4
            return (eval_at(2) - 4 * eval_at(1) + 6 * eval_at(0) - 4 * eval_at(-1) + eval_at(-2)) / (step_size^4)
        end
    end
    
    # Fallback to recursive central differences for mixed or higher-order derivatives
    if isempty(directions)
        return spec.value(args, spec.parameters)[1]
    else
        dir = first(directions)
        rest = directions[2:end]
        
        args_plus = [j == dir ? args[j] + step_size : args[j] for j in eachindex(args)]
        args_minus = [j == dir ? args[j] - step_size : args[j] for j in eachindex(args)]
        
        val_plus = _symbolic_derivative_fd(spec, args_plus, rest, ivs; ε = step_size)
        val_minus = _symbolic_derivative_fd(spec, args_minus, rest, ivs; ε = step_size)
        
        return (val_plus - val_minus) / (2 * step_size)
    end
end

"""
    _prewalk_substitute(expr, dv_ops, ivs, neural_specs; epsilon = nothing)

Single-pass prewalk substitution of dependent-variable calls in a symbolic expression.
Uses symbolic Finite Differences (via stencils) for all derivatives.
"""
function _prewalk_substitute(expr, dv_ops, ivs, neural_specs, integrand_info; epsilon::Union{Nothing, Real} = nothing)
    matcher = function (node)
        # --- Prewalk priority 1: Differential-wrapped DV chain ---
        deriv_info = _as_dv_derivative(node, dv_ops)
        if deriv_info !== nothing
            spec = neural_specs[deriv_info.dv_index]
            args = SymbolicUtils.arguments(deriv_info.call)
            directions = _derivative_directions(deriv_info.derivative_vars, ivs)
            replacement = _symbolic_derivative_fd(spec, args, directions, ivs; ε = epsilon)
            return Symbolics.unwrap(replacement)
        end

        # --- Prewalk priority 2: Bare DV call (no Differential wrapper) ---
        dv_index = _matching_dv_index(node, dv_ops)
        if dv_index !== nothing
            spec = neural_specs[dv_index]
            args = SymbolicUtils.arguments(node)
            replacement = spec.value(args, spec.parameters)[1]
            return Symbolics.unwrap(replacement)
        end

        # --- Prewalk priority 3: Symbolics.Integral call ---
        if SymbolicUtils.iscall(node) && SymbolicUtils.operation(node) isa Symbolics.Integral
            op = SymbolicUtils.operation(node)::Symbolics.Integral
            integrand_expr = SymbolicUtils.arguments(node)[1]
            
            integrating_var_indices = _integrating_variables(op.domain.variables, ivs)
            lb, ub = _get_limits(op.domain.domain)
            
            num_int_vars = length(integrating_var_indices)
            τs = collect(Symbolics.variables(:τ, 1:num_int_vars))
            sub_dict = Dict(ivs[integrating_var_indices[j]] => τs[j] for j in 1:num_int_vars)
            integrand_substituted = Symbolics.substitute(integrand_expr, sub_dict)
            
            # Recursively run _prewalk_substitute on the τ-renamed integrand before storing
            integrand_substituted = _prewalk_substitute(integrand_substituted, dv_ops, ivs, neural_specs, integrand_info; epsilon)
            
            id = length(integrand_info) + 1
            push!(integrand_info, SymbolicPINNIntegrandInfo(integrand_substituted, τs, lb, ub, integrating_var_indices))
            
            return SymbolicUtils.term(SymbolicPINNIntegralPlaceholder, id; type = Real, shape = SymbolicUtils.ShapeVecT())
        end

        return node
    end

    inbuilt_rewriter = SymbolicUtils.Rewriters.Prewalk(matcher)
    return inbuilt_rewriter(expr)
end


"""
    symbolic_pinn_residual(eq, ivs, dvs, neural_specs; epsilon = nothing)

Create a symbolic PINN residual for one ModelingToolkit equation by replacing dependent
variable calls and derivative calls with symbolic neural-network calls using a
single-pass prewalk substitution. Supports finite differences.
"""
function symbolic_pinn_residual(eq, ivs, dvs, neural_specs, eq_params = []; epsilon::Union{Nothing, Real} = nothing)
    raw = _equation_residual(eq)
    expr = Symbolics.unwrap(raw)
    dv_ops = _dv_operation.(dvs)
    
    clean_eq_params = (eq_params isa SciMLBase.NullParameters) ? () : eq_params
    
    integrand_info = SymbolicPINNIntegrandInfo[]
    substituted = _prewalk_substitute(expr, dv_ops, ivs, neural_specs, integrand_info; epsilon)
    
    integrand_syms = Symbolics.SymbolicT[]
    integrand_fns = Function[]
    if !isempty(integrand_info)
        num_integrals = length(integrand_info)
        integrand_syms = collect(Symbolics.variables(:integrand_fn, 1:num_integrals))
        integrand_fn_args = Symbolics.unwrap.(integrand_syms)
        
        iv_args = Symbolics.unwrap.(ivs)
        nn_args = map(spec -> spec.value, neural_specs)
        p_args = map(spec -> spec.parameters, neural_specs)
        eq_args = Symbolics.unwrap.(collect(clean_eq_params))
        
        for info in integrand_info
            all_integrand_build_args = vcat(
                info.τs,
                iv_args,
                nn_args,
                integrand_fn_args,
                p_args,
                eq_args
            )
            integrand_fn_expr = Symbolics.build_function(
                info.integrand_substituted,
                all_integrand_build_args...;
                expression = Val{true}
            )
            push!(integrand_fns, @RuntimeGeneratedFunction(integrand_fn_expr))
        end
        
        replace_matcher = function (node)
            if SymbolicUtils.iscall(node) && SymbolicUtils.operation(node) === SymbolicPINNIntegralPlaceholder
                id = SymbolicUtils.arguments(node)[1]
                idx = Int(Symbolics.value(id))
                info = integrand_info[idx]
                
                lb_args = Symbolics.unwrap.(info.lb)
                ub_args = Symbolics.unwrap.(info.ub)
                num_bounds = length(lb_args)
                
                call_args = vcat(
                    Any[integrand_syms[idx], num_bounds],
                    lb_args,
                    ub_args,
                    Any[length(ivs)],
                    iv_args,
                    nn_args,
                    integrand_fn_args,
                    p_args,
                    eq_args
                )
                return SymbolicUtils.term(_solve_pinn_integral, call_args...; type = Real, shape = SymbolicUtils.ShapeVecT())
            end
            return node
        end
        
        postwalk_rewriter = SymbolicUtils.Rewriters.Postwalk(replace_matcher)
        substituted = postwalk_rewriter(substituted)
    end
    
    return Num(substituted), integrand_syms, integrand_fns
end

function _contains_dv_call(expr, dvs)
    dv_ops = _dv_operation.(dvs)
    return SymbolicUtils.query(ex -> _matching_dv_index(ex, dv_ops) !== nothing, Symbolics.unwrap(expr))
end

function _theta0(spec::SymbolicPINNNeuralSpec)
    return Vector(Symbolics.getdefaultval(spec.parameters))
end

function _theta0(specs::AbstractVector{<:SymbolicPINNNeuralSpec})
    return reduce(vcat, [_theta0(spec) for spec in specs])
end

function _split_theta(theta, param_lengths)
    offsets = cumsum(param_lengths)
    return ntuple(Val(length(param_lengths))) do i
        lo = i == 1 ? 1 : offsets[i - 1] + 1
        @view(theta[lo:offsets[i]])
    end
end

# ---------- Pure BasicSymbolic residual compilation ----------

"""
    SymbolicPINNCompiledLoss{F, L, D, C, N_IV}

A compiled PINN loss function produced by the pure BasicSymbolic pipeline.

## Fields
- `scalar_fn`: Compiled scalar residual function `(x₁, x₂, ..., NN, p, ...) → scalar`.
  Produced by `Symbolics.build_function(expression=Val{false})` — no manual Expr editing.
- `nn_defaults`: Tuple of NN default value (Lux chain) closures.
- `integrand_fns`: Tuple of compiled integrand functions.
- `param_lengths`: Tuple of parameter vector lengths for theta splitting.
- `eq_param_count`: Val{C} — number of equation parameters.
- `default_eq_params`: Default equation parameter values (or nothing).
"""
struct SymbolicPINNCompiledLoss{F, MF, NN, IF, L, D, C, N_IV}
    scalar_fn::F
    mat_fn::MF
    nn_defaults::NN
    integrand_fns::IF
    param_lengths::L
    eq_param_count::Val{C}
    default_eq_params::D
end

function SymbolicPINNCompiledLoss(scalar_fn::F, mat_fn::MF, nn_defaults::NN, integrand_fns::IF, param_lengths::L,
        eq_param_count::Val{C}, default_eq_params::D, ::Val{N_IV}) where {F, MF, NN, IF, L, C, D, N_IV}
    return SymbolicPINNCompiledLoss{F, MF, NN, IF, L, D, C, N_IV}(
        scalar_fn, mat_fn, nn_defaults, integrand_fns, param_lengths, eq_param_count, default_eq_params)
end

_depvar_theta(theta) = hasproperty(theta, :depvar) ? theta.depvar : theta

function _eq_param_values(theta, ::Val{0}, default_eq_params)
    return ()
end

function _eq_param_values(theta, ::Val{C}, default_eq_params) where C
    if hasproperty(theta, :p)
        values = ntuple(i -> theta.p[i], Val(C))
    elseif default_eq_params !== nothing
        values = ntuple(i -> default_eq_params[i], Val(C))
    else
        throw(ArgumentError("Equation parameters are required but neither `theta.p` nor defaults were provided."))
    end
    return values
end

# Scalar evaluation: point is a vector
function (f::SymbolicPINNCompiledLoss{F, MF, NN, IF, L, D, C, N_IV})(point::AbstractVector, theta) where {F, MF, NN, IF, L, D, C, N_IV}
    depvar_theta = _depvar_theta(theta)
    param_views = _split_theta(depvar_theta, f.param_lengths)
    eq_values = _eq_param_values(theta, f.eq_param_count, f.default_eq_params)
    point_tuple = ntuple(d -> point[d], Val(N_IV))
    return f.scalar_fn(point_tuple..., f.nn_defaults..., f.integrand_fns..., param_views..., eq_values...)
end

# Batched evaluation: cord is a (D, N) matrix.
# Evaluates compiled function over collocation points vectorially.
function (f::SymbolicPINNCompiledLoss{F, MF, NN, IF, L, D, C, N_IV})(cord::AbstractMatrix, theta) where {F, MF, NN, IF, L, D, C, N_IV}
    isempty(cord) && return similar(cord, eltype(cord), 1, 0)
    depvar_theta = _depvar_theta(theta)
    param_views = _split_theta(depvar_theta, f.param_lengths)
    eq_values = _eq_param_values(theta, f.eq_param_count, f.default_eq_params)
    N_iv = Val(N_IV)
    row_inputs = ntuple(d -> d <= size(cord, 1) ? cord[d, :] : fill(zero(eltype(cord)), size(cord, 2)), N_iv)
    res = f.scalar_fn.(
        row_inputs...,
        Ref.(f.nn_defaults)...,
        Ref.(f.integrand_fns)...,
        Ref.(param_views)...,
        Ref.(eq_values)...
    )
    return res isa AbstractMatrix ? vec(res) : res
end

"""
    _compiled_residual(residual, ivs, neural_specs)

Compile a `BasicSymbolic` residual expression into an executable scalar and matrix function
using `Symbolics.build_function(expression=Val{false})`.
"""
function _compiled_residual(residual, ivs, neural_specs, integrand_syms = [], integrand_fns = [];
        eq_params = (), default_eq_params = nothing)
    iv_args = Symbolics.unwrap.(ivs)
    nn_args = map(spec -> spec.value, neural_specs)
    p_args = map(spec -> spec.parameters, neural_specs)
    clean_eq_params = (eq_params isa SciMLBase.NullParameters) ? () : eq_params
    eq_args = Symbolics.unwrap.(collect(clean_eq_params))
    integrand_fn_args = Symbolics.unwrap.(integrand_syms)

    # Build scalar compiled function via Symbolics standard codegen.
    all_build_args = vcat(iv_args, nn_args, integrand_fn_args, p_args, eq_args)
    scalar_fn = Symbolics.build_function(
        residual, all_build_args...;
        expression = Val{false}
    )

    # Build batched matrix evaluator for N collocation points if no integrals are present
    mat_fn = if isempty(integrand_syms) && !isempty(neural_specs)
        n_ivs = length(ivs)
        function (cord::AbstractMatrix, nn_defs::Tuple, integrand_defs::Tuple, params::Tuple, eq_vals::Tuple)
            N = size(cord, 2)
            isempty(cord) && return similar(cord, eltype(cord), 1, 0)
            
            # Evaluate Lux chains ONCE across all N collocation points in cord (D, N)
            nn_vals = ntuple(i -> begin
                chain = nn_defs[i]
                p = params[i]
                chain(cord, p)
            end, length(nn_defs))
            
            # Evaluate scalar_fn over row_inputs
            row_inputs = ntuple(d -> d <= size(cord, 1) ? cord[d, :] : fill(zero(eltype(cord)), N), Val(n_ivs))
            return scalar_fn.(
                row_inputs...,
                Ref.(nn_defs)...,
                Ref.(integrand_defs)...,
                Ref.(params)...,
                Ref.(eq_vals)...
            )
        end
    else
        nothing
    end

    # Bind NN default values (actual Lux chains) from the symbolic spec
    nn_defaults = Tuple(Symbolics.getdefaultval(spec.value) for spec in neural_specs)
    integrand_fn_tuple = Tuple(integrand_fns)

    param_lengths = Tuple(length(Symbolics.getdefaultval(spec.parameters)) for spec in neural_specs)

    return SymbolicPINNCompiledLoss(
        scalar_fn,
        mat_fn,
        nn_defaults,
        integrand_fn_tuple,
        param_lengths,
        Val(length(eq_args)),
        default_eq_params,
        Val(length(ivs))
    )
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
    axes = Tuple(_axis_points(lo, hi, n; interior) for (lo, hi) in _domain_bounds(domains))
    grid = vec([collect(Float64, point) for point in Iterators.product(axes...)])
    return reduce(hcat, grid)  # (D, N) matrix
end

function _find_dv_call(expr, dv_ops)
    found = Ref{Union{Symbolics.SymbolicT, Nothing}}(nothing)
    SymbolicUtils.query(Symbolics.unwrap(expr)) do ex
        if SymbolicUtils.iscall(ex) && any(dv_op -> isequal(SymbolicUtils.operation(ex), dv_op)::Bool, dv_ops)::Bool
            found[] = ex
            return true
        end
        return false
    end
    return found[]
end

function _bc_collocation_points(bc, ivs, dvs, domains, n_bc::Integer)
    dv_ops = _dv_operation.(dvs)
    dv_call = _find_dv_call(bc.lhs, dv_ops)
    if dv_call === nothing
        dv_call = _find_dv_call(bc.rhs, dv_ops)
    end
    
    if dv_call === nothing
        return _collocation_points(domains, n_bc; interior = false)
    end
    
    args = SymbolicUtils.arguments(Symbolics.unwrap(dv_call))
    axes_points = Vector{Float64}[]
    for (i, iv) in enumerate(ivs)
        arg = args[i]
        domain = domains[i].domain
        lo, hi = infimum(domain), supremum(domain)
        
        if isequal(arg, iv)
            pts = _axis_points(lo, hi, n_bc; interior = false)
            push!(axes_points, pts)
        else
            val = Float64(Symbolics.value(arg))
            push!(axes_points, [val])
        end
    end
    
    axes_points_tuple = Tuple(axes_points)
    grid = vec([collect(Float64, point) for point in Iterators.product(axes_points_tuple...)])
    return reduce(hcat, grid)  # (D, N) matrix
end

"""
    build_symbolic_pinn_ir(sys::PDESystem)

Construct a `SymbolicPINNIRStructure` containing pure `BasicSymbolic` residual ASTs
(`pde_residuals` and `bc_residuals`) without compiling early or approximating derivatives numerically.
"""
function build_symbolic_pinn_ir(sys::PDESystem)
    parsed = parse_pde_system(sys)
    pde_residuals = [Symbolics.unwrap(Symbolics.expand_derivatives(eq.lhs - eq.rhs)) for eq in parsed.eqs]
    bc_residuals  = [Symbolics.unwrap(Symbolics.expand_derivatives(bc.lhs - bc.rhs)) for bc in parsed.bcs]
    clean_ps = (parsed.ps isa SciMLBase.NullParameters) ? Num[] : collect(parsed.ps)
    return SymbolicPINNIRStructure(
        pde_residuals,
        bc_residuals,
        collect(parsed.ivs),
        collect(parsed.dvs),
        clean_ps,
        sys
    )
end

# Helper to extract a plain vector from compiled function output.
_to_residual_vector(x::AbstractVector) = x
_to_residual_vector(x::AbstractMatrix) = vec(x)
_to_residual_vector(x::Number) = [x]

"""
    _wrap_as_datafree(compiled_loss_fn)

Wrap a `SymbolicPINNCompiledLoss` into the `(cord::Matrix, θ) → Matrix` format
expected by NeuralPDE's training strategies (`GridTraining`, `StochasticTraining`, etc.).
"""
struct SymbolicPINNDatafreeLoss{F}
    res_fn::F
end

function (f::SymbolicPINNDatafreeLoss)(cord, theta)
    result = f.res_fn(cord, theta)
    return reshape(_to_residual_vector(result), 1, :)
end

function _wrap_as_datafree(compiled_loss_fn)
    return SymbolicPINNDatafreeLoss(compiled_loss_fn)
end

_weight_at(w::Number, i::Int) = w
_weight_at(w::Union{AbstractVector, Tuple}, i::Int) = w[i]

"""
    build_symbolic_pinn_loss(sys::PDESystem, chain; n_interior = 64, n_bc = 64,
                              pde_loss_weights = 1.0, bc_loss_weights = 1.0)

Build a symbolic PINN loss for a `PDESystem`. Supports single and multiple dependent
variables, same-direction and mixed/cross-direction derivatives. Supports optional fixed
or vector weights (`pde_loss_weights`, `bc_loss_weights`) for balancing loss terms.
The returned object is a named tuple containing symbolic residuals, lowered residual functions,
sampled points, initial parameters, and mean-squared PDE/BC/full loss functions.
"""
function build_symbolic_pinn_loss(sys::PDESystem, chain; n_interior::Integer = 64,
        n_bc::Integer = 64, epsilon::Union{Nothing, Real} = nothing,
        pde_loss_weights = 1.0, bc_loss_weights = 1.0)
    parsed = parse_pde_system(sys)

    n_inputs = [length(SymbolicUtils.arguments(Symbolics.unwrap(dv))) for dv in parsed.dvs]
    neural_specs = _symbolic_pinn_neural_specs(chain, n_inputs, length(parsed.dvs))
    theta0 = _theta0(neural_specs)

    pde_res_data = [
        symbolic_pinn_residual(eq, parsed.ivs, parsed.dvs, neural_specs, parsed.ps; epsilon)
            for eq in parsed.eqs
    ]
    pde_residuals = [x[1] for x in pde_res_data]
    pde_integrand_syms = [x[2] for x in pde_res_data]
    pde_integrand_fns = [x[3] for x in pde_res_data]

    bc_res_data = [
        symbolic_pinn_residual(bc, parsed.ivs, parsed.dvs, neural_specs, parsed.ps; epsilon)
            for bc in parsed.bcs
    ]
    bc_residuals = [x[1] for x in bc_res_data]
    bc_integrand_syms = [x[2] for x in bc_res_data]
    bc_integrand_fns = [x[3] for x in bc_res_data]

    # Compile residuals via Symbolics standard codegen (build_function expression=Val{false})
    # No manual Expr editing (_dot_pinn) — residuals stay BasicSymbolic until this point.
    pde_functions = [
        _compiled_residual(pde_residuals[i], parsed.ivs, neural_specs, pde_integrand_syms[i], pde_integrand_fns[i]; eq_params = parsed.ps)
            for i in 1:length(pde_residuals)
    ]
    bc_functions = [
        _compiled_residual(bc_residuals[i], parsed.ivs, neural_specs, bc_integrand_syms[i], bc_integrand_fns[i]; eq_params = parsed.ps)
            for i in 1:length(bc_residuals)
    ]

    datafree_pde_loss_functions = [_wrap_as_datafree(f) for f in pde_functions]
    datafree_bc_loss_functions = [_wrap_as_datafree(f) for f in bc_functions]

    pde_points = _collocation_points(parsed.domains, n_interior; interior = true)
    bc_points_list = [
        _bc_collocation_points(bc, parsed.ivs, parsed.dvs, parsed.domains, n_bc)
            for bc in parsed.bcs
    ]

    pde_loss = theta -> isempty(datafree_pde_loss_functions) ? zero(eltype(theta)) :
        sum(enumerate(datafree_pde_loss_functions)) do (i, f)
            _weight_at(pde_loss_weights, i) * mean(abs2, f(pde_points, theta))
        end / length(datafree_pde_loss_functions)

    bc_loss = theta -> isempty(datafree_bc_loss_functions) ? zero(eltype(theta)) :
        sum(enumerate(zip(datafree_bc_loss_functions, bc_points_list))) do (j, (f, pts))
            _weight_at(bc_loss_weights, j) * mean(abs2, f(pts, theta))
        end / length(datafree_bc_loss_functions)

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
        points = (pde = pde_points, bc = bc_points_list),
        pde_loss = pde_loss,
        bc_loss = bc_loss,
        loss = loss
    )
end


"""
    symbolic_pinn_loss_expression(sys::PDESystem, chain; epsilon = nothing)

Return the symbolic residual expressions for the PDE and boundary conditions, keeping
independent variables and network weights as symbols (no coordinates hardcoded).
"""
function symbolic_pinn_loss_expression(sys::PDESystem, chain; epsilon::Union{Nothing, Real} = nothing)
    loss_info = build_symbolic_pinn_loss(sys, chain; epsilon)
    return (
        pde = loss_info.pde_residuals,
        bc = loss_info.bc_residuals,
        ivs = loss_info.parsed.ivs,
        dvs = loss_info.parsed.dvs
    )
end
