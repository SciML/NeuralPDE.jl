# Lowering of PDESystem expressions into symbolic array expressions over a batch of
# collocation points. Every scalar independent variable becomes a `1 × n` row of the
# collocation matrix, dependent variables become batched network evaluations and scalar
# operations are broadcast, so the representation size is independent of `n`.

struct LoweringContext{X, I, G, N, P, D, T}
    xs::X                 # collocation matrix parameter (d × n) or `nothing`
    iv_index::I           # free independent variable => row of `xs`
    iv_global::G          # every independent variable => index into the shift vector
    networks::N           # depvar operation => TrialNetwork
    params::P             # PDESystem parameter => symbolic used in the System
    derivative::D
    npoints::Int
    eltype::Type{T}
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
        throw(
            ArgumentError(
                "`Integral` terms are not supported by the `PhysicsInformedNN` \
                discretization yet; see https://github.com/SciML/NeuralPDE.jl/issues/1161."
            )
        )
    elseif haskey(ctx.networks, op)
        return lower_depvar(ex, ctx, shift)
    end
    args = map(a -> lower(a, ctx, shift), arguments(ex))
    if any(_isarray, args)
        return Base.broadcast(op, args...)
    end
    return op(args...)
end

function lower_depvar(ex, ctx::LoweringContext, shift)
    net = ctx.networks[operation(ex)]
    callargs = arguments(ex)
    n_in = length(callargs)
    d = ctx.xs === nothing ? 0 : length(ctx.iv_index)
    P = zeros(ctx.eltype, n_in, d)
    c = zeros(ctx.eltype, n_in)
    for (j, a) in enumerate(callargs)
        a = unwrap(a)
        # A literal argument is still translated when differentiating with respect to the
        # variable of that slot: `Dx(u(1.0))` is the derivative of `u` evaluated at `x = 1`.
        if _isnumber(a)
            c[j] = _number(a)
            slot = unwrap(net.args[j])
            haskey(ctx.iv_global, slot) && (c[j] += shift[ctx.iv_global[slot]])
        elseif haskey(ctx.iv_index, a)
            P[j, ctx.iv_index[a]] = one(ctx.eltype)
            c[j] += shift[ctx.iv_global[a]]
        else
            throw(
                ArgumentError(
                    "Argument `$(a)` of `$(ex)` must be an independent variable or a \
                    number; general argument expressions are not supported yet."
                )
            )
        end
    end
    X = if ctx.xs === nothing
        reshape(c, n_in, 1)
    elseif P == I
        iszero(c) ? wrap(ctx.xs) : wrap(ctx.xs) .+ c
    else
        iszero(c) ? P * wrap(ctx.xs) : P * wrap(ctx.xs) .+ c
    end
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

function _collect_ivs!(found, ex, ivs)
    ex = unwrap(ex)
    if any(x -> isequal(unwrap(x), ex), ivs)
        push!(found, ex)
        return
    end
    iscall(ex) || return
    for a in arguments(ex)
        _collect_ivs!(found, a, ivs)
    end
    return
end

"""
    pinned_values(bcs, v::PDEBase.VariableMap)

Return a `Dict` mapping each independent variable to the set of numeric values it is
pinned to in the boundary conditions, e.g. `x => Set([0.0, 1.0])` for `u(0, y)` and
`u(1, y)`. `GridTraining` uses it to keep interior points off the boundaries.
"""
function pinned_values(bcs, v::PDEBase.VariableMap)
    pinned = Dict(unwrap(x) => Set{Float64}() for x in PDEBase.all_ivs(v))
    for bc in bcs, side in (bc.lhs, bc.rhs)
        _collect_pinned!(pinned, unwrap(side), v)
    end
    return pinned
end

function _collect_pinned!(pinned, ex, v)
    iscall(ex) || return
    op = operation(ex)
    if haskey(v.args, op)
        for (a, x) in zip(arguments(ex), v.args[op])
            a = unwrap(a)
            _isnumber(a) && haskey(pinned, unwrap(x)) &&
                push!(pinned[unwrap(x)], Float64(_number(a)))
        end
        return
    end
    for a in arguments(ex)
        _collect_pinned!(pinned, unwrap(a), v)
    end
    return
end
