# Lowering of PDESystem expressions into symbolic array expressions over a batch of
# collocation points. Every scalar independent variable becomes a `1 × n` row of the
# collocation matrix, dependent variables become batched network evaluations and scalar
# operations are broadcast, so the representation size is independent of `n`.

struct LoweringContext{X, I, G, N, P, D, T, E}
    xs::X                 # collocation matrix parameter (d × n) or `nothing`
    iv_index::I           # free independent variable => row of `xs`
    iv_global::G          # every independent variable => index into the shift vector
    networks::N           # depvar operation => TrialNetwork
    params::P             # PDESystem parameter => symbolic used in the System
    derivative::D
    npoints::Int
    eltype::Type{T}
    integral_alg::E       # fixed-node rule for `Integral` terms
    extras::Vector{Any}   # parameters created by integral lowering (qf, ξ, w)
    tag::Symbol           # block tag used to name created parameters
    # Shift at the entry of the innermost enclosing general-call argument, or `nothing`.
    # Plain-call literals move only by `shift - literal_base`, so an outer finite
    # difference of `u(u(0.0, y), 2x)` does not move the inner `0.0`.
    literal_base::Union{Nothing, Vector{T}}
end

function LoweringContext(
        xs, iv_index, iv_global, networks, params, derivative, npoints, ::Type{T},
        integral_alg, extras, tag, literal_base = nothing
    ) where {T}
    return LoweringContext{
        typeof(xs), typeof(iv_index), typeof(iv_global),
        typeof(networks), typeof(params), typeof(derivative), T, typeof(integral_alg),
    }(
        xs, iv_index, iv_global, networks, params, derivative, npoints, T,
        integral_alg, extras, tag, literal_base
    )
end

function _with_literal_base(ctx::LoweringContext, shift)
    return LoweringContext(
        ctx.xs, ctx.iv_index, ctx.iv_global, ctx.networks, ctx.params, ctx.derivative,
        ctx.npoints, ctx.eltype, ctx.integral_alg, ctx.extras, ctx.tag,
        collect(ctx.eltype, shift)
    )
end

_isnumber(ex) = ex isa Number || (ex isa SymbolicUtils.BasicSymbolic && SymbolicUtils.isconst(ex))
_number(ex) = ex isa Number ? ex : SymbolicUtils.unwrap_const(ex)

# Non-integer literals take the parameter element type so that Float32 networks stay Float32.
_literal(x::Integer, ::Type{T}) where {T} = x
_literal(x::Real, ::Type{T}) where {T} = T(x)
_literal(x, ::Type{T}) where {T} = x

_isarray(ex) = ex isa Symbolics.Arr || (ex isa SymbolicUtils.BasicSymbolic && SymbolicUtils.symtype(ex) <: AbstractArray)

"""
    lower(ex, ctx::LoweringContext, shift)

Lower the scalar expression `ex` into a `Num` (when it does not depend on the collocation
points) or a `1 × n` `Arr` evaluated at the collocation points translated by `shift`.
`shift` has one entry per independent variable of the `PDESystem` (see
`LoweringContext.iv_global`); finite-difference stencils are built by translating it.
"""
function lower(ex, ctx::LoweringContext, shift)
    ex = unwrap(ex)
    _isnumber(ex) && return _literal(_number(ex), ctx.eltype)
    if haskey(ctx.iv_index, ex)
        row = ctx.iv_index[ex]
        r = wrap(ctx.xs)[row:row, :]
        s = shift[ctx.iv_global[ex]]
        return iszero(s) ? r : r .+ s
    end
    haskey(ctx.params, ex) && return ctx.params[ex]
    iscall(ex) || return wrap(ex)
    op = operation(ex)
    if op isa Differential
        return lower_differential(ex, ctx, shift)
    elseif op isa Symbolics.Integral
        return lower_integral(ex, ctx, shift)
    elseif haskey(ctx.networks, op)
        return lower_depvar(ex, ctx, shift)
    end
    args = map(a -> lower(a, ctx, shift), arguments(ex))
    if any(_isarray, args)
        return Base.broadcast(op, args...)
    end
    return op(args...)
end

function _is_general_call(callargs, ctx::LoweringContext)
    for a in callargs
        a = unwrap(a)
        (_isnumber(a) || haskey(ctx.iv_index, a)) || return true
    end
    return false
end

# Whether `ex` depends on the independent variable `x` under the lowering's semantics:
# a plain call's literal in the `x` slot counts (boundary convention) unless the call
# sits inside a general argument (`fixed`), where literals do not move. Used to reject
# vacuous derivatives such as `Dx(u(0.0, 2y))`.
function _composition_depends_on_iv(ex, x, ctx::LoweringContext, fixed = false)
    ex = unwrap(ex)
    isequal(ex, x) && return true
    iscall(ex) || return false
    op = operation(ex)
    if op isa Symbolics.Integral
        ivars = _integral_variables(op.domain)
        lbs, ubs = _integral_bounds(op.domain.domain)
        any(b -> _composition_depends_on_iv(b, x, ctx, fixed), (lbs..., ubs...)) &&
            return true
        any(v -> isequal(unwrap(v), x), ivars) && return false
        return _composition_depends_on_iv(only(arguments(ex)), x, ctx, true)
    elseif haskey(ctx.networks, op)
        callargs = arguments(ex)
        general = _is_general_call(callargs, ctx)
        for (j, a) in enumerate(callargs)
            a = unwrap(a)
            if _isnumber(a)
                !general && !fixed && isequal(unwrap(ctx.networks[op].args[j]), x) &&
                    return true
            elseif _composition_depends_on_iv(a, x, ctx, fixed || general)
                return true
            end
        end
        return false
    end
    return any(a -> _composition_depends_on_iv(a, x, ctx, fixed), arguments(ex))
end

function _involves_general_call(ex, ctx::LoweringContext)
    ex = unwrap(ex)
    iscall(ex) || return false
    op = operation(ex)
    if haskey(ctx.networks, op)
        _is_general_call(arguments(ex), ctx) && return true
    end
    return any(a -> _involves_general_call(a, ctx), arguments(ex))
end

function lower_depvar(ex, ctx::LoweringContext, shift, directions = nothing)
    net = ctx.networks[operation(ex)]
    callargs = arguments(ex)
    n_in = length(callargs)
    # `iv_index` values are rows of `ctx.xs`; an integrating variable that shadows an
    # outer variable name replaces its entry, so the row count is the largest value.
    d = ctx.xs === nothing ? 0 : maximum(values(ctx.iv_index))
    P = zeros(ctx.eltype, n_in, d)
    c = zeros(ctx.eltype, n_in)
    general = _is_general_call(callargs, ctx)
    for (j, a) in enumerate(callargs)
        a = unwrap(a)
        if _isnumber(a)
            c[j] = _number(a)
        elseif haskey(ctx.iv_index, a)
            P[j, ctx.iv_index[a]] = one(ctx.eltype)
            c[j] += shift[ctx.iv_global[a]]
        end
    end
    # Plain-call boundary convention: `Dx(u(1.0))` is ∂u/∂x at the pinned point, so the
    # finite-difference shift is applied to the literal. In a general composition such as
    # `Dx(u(0.0, 2x))` the derivative is of the map `x ↦ u(0.0, 2x)` and literals stay fixed;
    # only free-coordinate / general argument rows pick up the shift (via `lower` below).
    if !general
        lshift = ctx.literal_base === nothing ? shift : shift .- ctx.literal_base
        for (j, a) in enumerate(callargs)
            a = unwrap(a)
            if _isnumber(a)
                slot = unwrap(net.args[j])
                haskey(ctx.iv_global, slot) && (c[j] += lshift[ctx.iv_global[slot]])
            end
        end
    end
    X = if general
        # A general argument such as `u(t - τ)` lowers to its own `1 × n` row; the rows
        # are stacked lazily so the expression does not scalarize over the batch.
        ctx_args = _with_literal_base(ctx, shift)
        rows = map(enumerate(callargs)) do (j, a)
            a = unwrap(a)
            if _isnumber(a)
                wrap(fill(ctx.eltype(0), 1, ctx.npoints))
            elseif haskey(ctx.iv_index, a)
                wrap(ctx.xs)[ctx.iv_index[a]:ctx.iv_index[a], :]
            else
                r = lower(a, ctx_args, shift)
                _isarray(r) ? r : wrap(fill(unwrap(r), 1, ctx.npoints))
            end
        end
        Xg = foldl(nn_vcat, rows)
        iszero(c) ? Xg : Xg .+ reshape(c, n_in, 1)
    elseif ctx.xs === nothing
        reshape(c, n_in, 1)
    elseif P == I
        iszero(c) ? wrap(ctx.xs) : wrap(ctx.xs) .+ c
    else
        iszero(c) ? P * wrap(ctx.xs) : P * wrap(ctx.xs) .+ c
    end
    directions === nothing || return nn_jvp(net.NN, X, net.θ, directions, net.output)
    net.noutputs == 1 && return nn_eval(net.NN, X, net.θ)
    return nn_eval_row(net.NN, X, net.θ, net.output)
end

function lower_differential(ex, ctx::LoweringContext, shift)
    D = operation(ex)
    x = D.x
    inner = arguments(ex)[1]
    D.order isa Integer || throw(
        ArgumentError("Fractional derivative orders are not supported, got `$(D)`.")
    )
    haskey(ctx.iv_global, unwrap(x)) || throw(
        ArgumentError(
            "Differential with respect to `$(x)`, which is not an independent variable of \
            the `PDESystem`."
        )
    )
    # A general call keeps literal slots fixed, so `Dx(u(0.0, 2y))` would lower to the
    # zero map with no dependence on θ. Reject that instead of a silent constant residual.
    if _involves_general_call(inner, ctx) &&
            !_composition_depends_on_iv(inner, unwrap(x), ctx)
        throw(
            ArgumentError(
                "Differentiating `$(inner)` with respect to `$(x)` has no effect under \
                composition semantics: `$(x)` appears only as a fixed literal in a \
                general-argument call. Use a plain call such as `u(0.0, y)` for a boundary \
                derivative at a pinned coordinate, or include `$(x)` in a non-literal \
                argument (for example `Dx(u(0.0, 2x))`)."
            )
        )
    end
    direction = ctx.iv_global[unwrap(x)]
    order = D.order
    # Orders above four are lowered as a first-order stencil of the next lower order.
    if order > 4
        inner = Differential(x, order - 1)(inner)
        order = 1
    end
    return lower_derivative(ctx.derivative, inner, ctx, shift, direction, order)
end

"""
    lower_derivative(lowering, inner, ctx, shift, direction, order)

Return the symbolic expression for the `order`-th derivative of `inner` along the
independent variable with global index `direction`, evaluated at the collocation points
translated by `shift`.
"""
function lower_derivative(
        ::FiniteDifferenceDerivative, inner, ctx::LoweringContext, shift, direction, order
    )
    T = ctx.eltype
    ε = eps(T)^(one(T) / (2 + order))
    e = zeros(T, length(shift))
    e[direction] = one(T)
    L(k) = lower(inner, ctx, shift .+ (k * ε) .* e)
    if order == 1
        return (L(1) .- L(-1)) ./ (2ε)
    elseif order == 2
        return (L(1) .+ L(-1) .- 2 .* L(0)) ./ ε^2
    elseif order == 3
        return (L(2) .- 2 .* L(1) .+ 2 .* L(-1) .- L(-2)) ./ (2ε^3)
    else
        return (L(2) .- 4 .* L(1) .+ 6 .* L(0) .- 4 .* L(-1) .+ L(-2)) ./ ε^4
    end
end

"""
    free_ivs(ex, ivs, depvar_ops)

Return the independent variables of `ivs` that `ex` depends on, either directly or
through the arguments of a dependent variable call, in the order of `ivs`. The variable
of a `Differential` is not free by itself: `Dx(u(1.0))` is a point condition.
"""
function free_ivs(ex, ivs, depvar_ops)
    found = Set{Any}()
    _collect_ivs!(found, unwrap(ex), ivs)
    return filter(x -> unwrap(x) in found, ivs)
end

function _collect_ivs!(found, ex, ivs, exclude = ())
    ex = unwrap(ex)
    any(v -> isequal(unwrap(v), ex), exclude) && return
    if any(x -> isequal(unwrap(x), ex), ivs)
        push!(found, ex)
        return
    end
    iscall(ex) || return
    op = operation(ex)
    if op isa Symbolics.Integral
        # Free variables can hide in the domain bounds. Integrating variables are
        # bound: inside the integrand they are all excluded, and in the bound of the
        # `i`-th variable only the variables `1:i-1` may be bound (intervals compose
        # lexicographically); a later integrating-variable name there is an outer
        # variable that happens to share the symbol.
        ivars = _integral_variables(op.domain)
        lbs, ubs = _integral_bounds(op.domain.domain)
        for i in eachindex(ivars)
            for b in (lbs[i], ubs[i])
                _collect_ivs!(found, b, ivs, (exclude..., ivars[1:(i - 1)]...))
            end
        end
        for a in arguments(ex)
            _collect_ivs!(found, a, ivs, (exclude..., ivars...))
        end
        return
    end
    for a in arguments(ex)
        _collect_ivs!(found, a, ivs, exclude)
    end
    return
end

"""
    pinned_values(bcs, pdesys)

Return a `Dict` mapping each independent variable to the set of numeric values it is
pinned to in the boundary conditions, e.g. `x => Set([0.0, 1.0])` for `u(0, y)` and
`u(1, y)`. `GridTraining` uses it to keep interior points off the boundaries.

The declared signatures of `get_dvs(pdesys)` are used as the argument slots, not
`VariableMap.args`, so a call with a general argument (`u(t, x + 1)`) does not
overwrite the pin mapping for `u(t, 0)`.
"""
function pinned_values(bcs, pdesys)
    pinned = Dict(unwrap(x) => Set{Float64}() for x in get_ivs(pdesys))
    declared = Dict(
        unwrap(operation(dv)) => Any[unwrap(a) for a in arguments(dv)]
            for dv in get_dvs(pdesys)
    )
    for bc in bcs, side in (bc.lhs, bc.rhs)
        _collect_pinned!(pinned, unwrap(side), declared)
    end
    return pinned
end

function _collect_pinned!(pinned, ex, declared)
    iscall(ex) || return
    op = unwrap(operation(ex))
    if haskey(declared, op)
        for (a, x) in zip(arguments(ex), declared[op])
            a = unwrap(a)
            _isnumber(a) && haskey(pinned, unwrap(x)) &&
                push!(pinned[unwrap(x)], Float64(_number(a)))
        end
        return
    end
    for a in arguments(ex)
        _collect_pinned!(pinned, unwrap(a), declared)
    end
    return
end
