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
struct BatchVector{T}
    data::Vector{T}
end
Base.getindex(x::BatchVector, i::Integer) = x.data

function ChainRulesCore.rrule(::typeof(Base.getindex), x::BatchVector, i::Integer)
    val = x.data
    function getindex_pullback(Δ)
        return NoTangent(), ChainRulesCore.Tangent{BatchVector}(data = Δ), NoTangent()
    end
    return val, getindex_pullback
end

Base.length(x::BatchVector) = length(x.data)
Base.size(x::BatchVector) = size(x.data)
Base.Broadcast.broadcastable(x::BatchVector) = x.data

function ChainRulesCore.rrule(::Type{BatchVector}, data::AbstractVector)
    val = BatchVector(data)
    function BatchVector_pullback(Δ)
        Δ_data = if Δ isa ChainRulesCore.Tangent
            Δ.data isa NoTangent ? zero(data) : Δ.data
        elseif hasproperty(Δ, :data)
            Δ.data
        else
            Δ
        end
        return NoTangent(), Δ_data
    end
    return val, BatchVector_pullback
end

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
    elseif SymbolicUtils.iscall(unwrapped_vars) && (SymbolicUtils.operation(unwrapped_vars) === tuple || SymbolicUtils.operation(unwrapped_vars) === Symbolics.tuple)
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

function _get_value_at(x::AbstractVector, i)
    return x[i]
end
function _get_value_at(x::BatchVector, i)
    return x.data[i]
end
function _get_value_at(x, i)
    return x
end

function _solve_pinn_integral(integrand_fn, num_bounds::Int, rest...)
    idx_num_ivs = 2 * num_bounds + 1
    num_ivs = rest[idx_num_ivs]
    args = rest[(idx_num_ivs + 1):end]

    is_batch = false
    N = 1
    for j in 1:num_ivs
        val = args[j]
        if val isa AbstractVector
            is_batch = true
            N = length(val)
            break
        end
    end
    for j in 1:(2*num_bounds)
        val = rest[j]
        if val isa AbstractVector && length(val) > 1
            is_batch = true
            N = length(val)
            break
        end
    end

    if is_batch
        results = map(1:N) do i
            point_i = ntuple(j -> j <= num_ivs ? _get_value_at(args[j], i) : args[j], length(args))
            if num_bounds == 1
                lb_val = _get_value_at(rest[1], i)
                ub_val = _get_value_at(rest[2], i)
            else
                lb_val = [_get_value_at(rest[j], i) for j in 1:num_bounds]
                ub_val = [_get_value_at(rest[num_bounds + j], i) for j in 1:num_bounds]
            end
            
            integrand = if num_bounds > 1
                (τ, p_) -> integrand_fn(τ..., point_i...)
            else
                (τ, p_) -> integrand_fn(τ, point_i...)
            end
            prob = Integrals.IntegralProblem(integrand, (lb_val, ub_val))
            sol = Integrals.solve(prob, Integrals.CubatureJLh(), reltol = 1e-3, abstol = 1e-3)
            sol.u
        end
        return BatchVector(results)
    else
        if num_bounds == 1
            lb_val = rest[1]
            ub_val = rest[2]
        else
            lb_val = [rest[j] for j in 1:num_bounds]
            ub_val = [rest[num_bounds + j] for j in 1:num_bounds]
        end
        
        integrand = if num_bounds > 1
            (τ, p_) -> integrand_fn(τ..., args...)
        else
            (τ, p_) -> integrand_fn(τ, args...)
        end
        prob = Integrals.IntegralProblem(integrand, (lb_val, ub_val))
        sol = Integrals.solve(prob, Integrals.CubatureJLh(), reltol = 1e-3, abstol = 1e-3)
        return [sol.u]
    end
end

struct SymbolicPINNValueWrapper{F}
    nn::F
end
Base.Broadcast.broadcastable(x::SymbolicPINNValueWrapper) = Ref(x)

function (f::SymbolicPINNValueWrapper)(input::AbstractVector, p)
    if any(x -> x isa AbstractVector, input)
        # Batch evaluation
        idx = findfirst(x -> x isa AbstractVector, input)
        N = length(input[idx])
        D = length(input)
        T = eltype(p)
        for x in input
            T = promote_type(T, x isa AbstractVector ? eltype(x) : typeof(x))
        end
        mat = Matrix{T}(undef, D, N)
        for i in 1:D
            mat[i, :] .= input[i]
        end
        out = f.nn(mat, p)
        return BatchVector(vec(out))
    else
        return f.nn(input, p)
    end
end

function ChainRulesCore.rrule(f::SymbolicPINNValueWrapper, input::AbstractVector, p)
    if any(x -> x isa AbstractVector, input)
        # Batch evaluation
        idx = findfirst(x -> x isa AbstractVector, input)
        N = length(input[idx])
        D = length(input)
        T = eltype(p)
        for x in input
            T = promote_type(T, x isa AbstractVector ? eltype(x) : typeof(x))
        end
        mat = Matrix{T}(undef, D, N)
        for i in 1:D
            mat[i, :] .= input[i]
        end
        out, nn_pullback = Zygote.pullback(p_ -> f.nn(mat, p_), p)
        val = BatchVector(vec(out))
        
        function SymbolicPINNValueWrapper_pullback(Δ)
            Δ_data = if Δ isa ChainRulesCore.Tangent
                Δ.data isa NoTangent ? zero(vec(out)) : Δ.data
            elseif hasproperty(Δ, :data)
                Δ.data
            else
                Δ
            end
            Δ_mat = reshape(Δ_data, 1, N)
            d_p = only(nn_pullback(Δ_mat))
            return NoTangent(), NoTangent(), d_p
        end
        return val, SymbolicPINNValueWrapper_pullback
    else
        val, nn_pullback = Zygote.pullback(p_ -> f.nn(input, p_), p)
        function SymbolicPINNValueWrapper_pullback_scalar(Δ)
            d_p = only(nn_pullback(Δ))
            return NoTangent(), NoTangent(), d_p
        end
        return val, SymbolicPINNValueWrapper_pullback_scalar
    end
end

function _replace_index(x::AbstractVector, i::Integer, val)
    T = promote_type(typeof(val), eltype(x))
    return T[j == i ? val : x[j] for j in eachindex(x)]
end

function ChainRulesCore.rrule(::typeof(_replace_index), x::AbstractVector, i::Integer, val)
    y = _replace_index(x, i, val)
    function _replace_index_pullback(Δ)
        T = promote_type(eltype(Δ), typeof(zero(val)))
        dx = T[j == i ? zero(Δ[j]) : Δ[j] for j in eachindex(x)]
        dval = Δ[i]
        return NoTangent(), dx, NoTangent(), dval
    end
    return y, _replace_index_pullback
end

function _replace_index_matrix(X::AbstractMatrix, col::Integer, z::AbstractVector)
    T = promote_type(eltype(z), eltype(X))
    out = Matrix{T}(undef, size(X)...)
    copyto!(out, X)
    out[:, col] .= z
    return out
end

# ---------- Broadcasting transformation for batched evaluation ----------

"""
    _is_pinn_dottable(fn_expr, skip_set::Set{Symbol})

Check whether a function call in the compiled PINN expression should be broadcasted.
Returns `false` for NN/DNN wrapper calls, `getindex`, and array constructors.
Returns `true` for arithmetic and math functions (`+`, `-`, `sin`, etc.).
"""
function _is_pinn_dottable(fn_expr, skip_set::Set{Symbol})
    name = if fn_expr isa Symbol
        fn_expr
    elseif fn_expr isa Function
        nameof(fn_expr)
    elseif fn_expr isa GlobalRef
        fn_expr.name
    elseif fn_expr isa Expr && fn_expr.head === :. && length(fn_expr.args) == 2
        # Module-qualified name, e.g. NaNMath.sin → extract :sin
        fn_expr.args[2] isa QuoteNode ? fn_expr.args[2].value : nothing
    else
        nothing
    end
    name === nothing && return false
    name === :getindex && return false
    name === :vect && return false
    name === :array_literal && return false
    name === :_solve_pinn_integral && return false
    name in skip_set && return false
    return true
end

"""
    _dot_pinn(x::Expr, skip_set::Set{Symbol})

Walk a Julia `Expr` tree and broadcast scalar function calls (`.+`, `sin.()`, etc.),
while skipping NN/DNN wrapper calls, `getindex`, and array constructors.
This allows the compiled residual function to operate on batched row-vector inputs.
"""
function _dot_pinn(x::Expr, skip_set::Set{Symbol})
    dotargs = Any[_dot_pinn(a, skip_set) for a in x.args]
    if x.head === :call && _is_pinn_dottable(x.args[1], skip_set)
        return Expr(:., dotargs[1], Expr(:tuple, dotargs[2:end]...))
    end
    return Expr(x.head, dotargs...)
end
_dot_pinn(x, ::Set{Symbol}) = x

"""
    _extract_arg_names(fn_expr::Expr)

Extract argument symbol names from a function `Expr` returned by
`Symbolics.build_function(expression = Val(true))`.
"""
function _extract_arg_names(fn_expr::Expr)
    if fn_expr.head === :function
        sig = fn_expr.args[1]
        if sig isa Expr && sig.head === :call
            return Symbol[_arg_name(a) for a in sig.args[2:end]]
        elseif sig isa Expr && sig.head === :tuple
            return Symbol[_arg_name(a) for a in sig.args]
        end
    elseif fn_expr.head === :->
        lhs = fn_expr.args[1]
        if lhs isa Expr && lhs.head === :tuple
            return Symbol[_arg_name(a) for a in lhs.args]
        elseif lhs isa Symbol
            return Symbol[lhs]
        end
    end
    return Symbol[]
end

function _arg_name(a)
    a isa Symbol && return a
    if a isa Expr && a.head === :(::)
        return _arg_name(a.args[1])
    end
    return :_unknown_arg
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
    return findfirst(dv_op -> isequal(op, dv_op), dv_ops)
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
            push!(integrand_info, (; integrand_substituted, τs, lb, ub, integrating_var_indices))
            
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
    
    integrand_info = []
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
            integrand_fn_expr = Symbolics.build_function(
                info.integrand_substituted,
                info.τs...,
                iv_args...,
                nn_args...,
                integrand_fn_args...,
                p_args...,
                eq_args...;
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
                
                call_args = Any[
                    integrand_syms[idx],
                    num_bounds,
                    lb_args...,
                    ub_args...,
                    length(ivs),
                    iv_args...,
                    nn_args...,
                    integrand_fn_args...,
                    p_args...,
                    eq_args...
                ]
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
    return ntuple(length(param_lengths)) do i
        lo = i == 1 ? 1 : offsets[i - 1] + 1
        @view(theta[lo:offsets[i]])
    end
end

function _runtime_args(neural_specs)
    nn_defaults = map(spec -> SymbolicPINNValueWrapper(Symbolics.getdefaultval(spec.value)), neural_specs)
    return (nn_defaults...,)
end

struct SymbolicPINNResidualFunction{F, R, L, D, C, N_IV}
    compiled::F
    runtime_args::R
    param_lengths::L
    eq_param_count::Val{C}
    default_eq_params::D
end

function SymbolicPINNResidualFunction(compiled::F, runtime_args::R, param_lengths::L, eq_param_count::Val{C}, default_eq_params::D, ::Val{N_IV}) where {F, R, L, C, D, N_IV}
    return SymbolicPINNResidualFunction{F, R, L, D, C, N_IV}(compiled, runtime_args, param_lengths, eq_param_count, default_eq_params)
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
function (f::SymbolicPINNResidualFunction{F, R, L, D, C, N_IV})(point::AbstractVector, theta) where {F, R, L, D, C, N_IV}
    depvar_theta = _depvar_theta(theta)
    param_views = _split_theta(depvar_theta, f.param_lengths)
    eq_values = _eq_param_values(theta, f.eq_param_count, f.default_eq_params)
    point_tuple = ntuple(d -> point[d], Val(N_IV))
    return f.compiled(point_tuple..., f.runtime_args..., param_views..., eq_values...)
end

# Batched evaluation: cord is a (D, N) matrix.
# Splits the coordinate matrix into row vectors and passes them to the
# broadcasted compiled function, so that dotted arithmetic and batched
# NN wrappers operate on the full batch in a single call.
function (f::SymbolicPINNResidualFunction{F, R, L, D, C, N_IV})(cord::AbstractMatrix, theta) where {F, R, L, D, C, N_IV}
    isempty(cord) && return similar(cord, eltype(cord), 1, 0)
    depvar_theta = _depvar_theta(theta)
    param_views = _split_theta(depvar_theta, f.param_lengths)
    eq_values = _eq_param_values(theta, f.eq_param_count, f.default_eq_params)
    n_points = size(cord, 2)
    row_inputs = ntuple(d -> d <= size(cord, 1) ? cord[d, :] : fill(zero(eltype(cord)), n_points), Val(N_IV))
    return f.compiled(row_inputs..., f.runtime_args..., param_views..., eq_values...)
end

"""
    _compiled_residual(residual, ivs, neural_specs)

Lower a symbolic residual expression into an executable function using
`Symbolics.build_function`. The generated expression is transformed with
`_dot_pinn` to broadcast arithmetic operations (`.+`, `sin.()`, etc.)
while preserving NN wrapper calls, then compiled with
`@RuntimeGeneratedFunction`. This enables batched evaluation on row-vector
inputs from the `(D, N)` coordinate matrix.
"""
function _compiled_residual(residual, ivs, neural_specs, integrand_syms = [], integrand_fns = [];
        eq_params = (), default_eq_params = nothing)
    iv_args = Symbolics.unwrap.(ivs)
    nn_args = map(spec -> spec.value, neural_specs)
    p_args = map(spec -> spec.parameters, neural_specs)
    clean_eq_params = (eq_params isa SciMLBase.NullParameters) ? () : eq_params
    eq_args = Symbolics.unwrap.(collect(clean_eq_params))
    integrand_fn_args = Symbolics.unwrap.(integrand_syms)

    # Get expression form for broadcasting transformation
    # We include integrand_fn_args in the build_function arguments!
    fn_expr = Symbolics.build_function(
        residual, iv_args..., nn_args..., integrand_fn_args..., p_args..., eq_args...;
        expression = Val{true}
    )

    # Build skip set: all argument names after the independent variables
    # (NN wrappers, integrand functions, and parameter vectors must not be broadcasted)
    arg_names = _extract_arg_names(fn_expr)
    n_iv = length(iv_args)
    skip_set = Set{Symbol}(arg_names[(n_iv + 1):end])

    # Apply broadcasting transformation and compile
    dotted_fn = _dot_pinn(fn_expr, skip_set)
    compiled = @RuntimeGeneratedFunction(dotted_fn)

    nn_defaults = map(spec -> SymbolicPINNValueWrapper(Symbolics.getdefaultval(spec.value)), neural_specs)
    runtime_args = (nn_defaults..., integrand_fns...)
    
    param_lengths = Tuple(length(Symbolics.getdefaultval(spec.parameters)) for spec in neural_specs)

    return SymbolicPINNResidualFunction(
        compiled,
        runtime_args,
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
    found = Ref{Any}(nothing)
    SymbolicUtils.query(Symbolics.unwrap(expr)) do ex
        if SymbolicUtils.iscall(ex) && any(dv_op -> isequal(SymbolicUtils.operation(ex), dv_op), dv_ops)
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

# Helper to extract a plain vector from batched compiled function output.
_to_residual_vector(x::AbstractVector) = x
_to_residual_vector(x::AbstractMatrix) = vec(x)
_to_residual_vector(x::Number) = [x]
_to_residual_vector(x::BatchVector) = x.data

"""
    _wrap_as_datafree(compiled_residual_fn)

Wrap a compiled residual function into the `(cord::Matrix, θ) -> Matrix` format
expected by NeuralPDE's training strategies (`GridTraining`, `StochasticTraining`, etc.).

The returned function accepts a `(D, N)` coordinate matrix and a parameter vector,
and returns a `(1, N)` matrix of residual values, compatible with
`merge_strategy_with_loss_function` and `get_loss_function`.
"""
struct SymbolicPINNDatafreeLoss{F}
    res_fn::F
end

function (f::SymbolicPINNDatafreeLoss)(cord, theta)
    result = f.res_fn(cord, theta)
    return reshape(_to_residual_vector(result), 1, :)
end

function _wrap_as_datafree(compiled_residual_fn)
    return SymbolicPINNDatafreeLoss(compiled_residual_fn)
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
        n_bc::Integer = 64, epsilon::Union{Nothing, Real} = nothing)
    parsed = parse_pde_system(sys)

    neural_specs = _symbolic_pinn_neural_specs(chain, length(parsed.ivs), length(parsed.dvs))
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

    pde_loss = theta -> _mean_square(datafree_pde_loss_functions, pde_points, theta)
    bc_loss = theta -> sum(zip(datafree_bc_loss_functions, bc_points_list)) do (f, pts)
        mean(abs2, f(pts, theta))
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
