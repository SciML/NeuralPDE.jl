_network_jvp(f, X, θ, ::Val{()}) = f(X, θ)

function _network_jvp(f, X, θ, ::Val{D}) where {D}
    dX = Enzyme.make_zero(X)
    for row in first(D)
        dX[row:row, :] .= one(eltype(X))
    end
    # Zero parameter tangents avoid mixed-activity buffers in fused network operations.
    return only(
        Enzyme.autodiff(
            Enzyme.Forward, _network_jvp, Enzyme.Duplicated,
            Enzyme.Const(f), Enzyme.Duplicated(X, dX), Enzyme.Duplicated(θ, Enzyme.make_zero(θ)),
            Enzyme.Const(Val(Base.tail(D)))
        )
    )
end

nn_jvp(f, X, θ, directions, k) = _network_jvp(f, X, θ, directions)[k:k, :]

function _jvp_contraction(f, X, θ, directions, k, Δ)
    return sum(nn_jvp(f, X, θ, directions, k) .* Δ)
end

function ChainRulesCore.rrule(::typeof(nn_jvp), f, X, θ, directions, k)
    Y = nn_jvp(f, X, θ, directions, k)
    function jvp_pullback(ΔY)
        Δ = ChainRulesCore.unthunk(ΔY)
        if Δ isa ChainRulesCore.AbstractZero
            return (
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
                ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(),
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
            )
        end
        dX, dθ = Enzyme.make_zero(X), Enzyme.make_zero(θ)
        Enzyme.autodiff(
            Enzyme.Reverse, _jvp_contraction, Enzyme.Active, Enzyme.Const(f),
            Enzyme.Duplicated(X, dX), Enzyme.Duplicated(θ, dθ),
            Enzyme.Const(directions), Enzyme.Const(k), Enzyme.Const(Δ)
        )
        return (
            ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
            ChainRulesCore.ProjectTo(X)(dX), ChainRulesCore.ProjectTo(θ)(dθ),
            ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
        )
    end
    return Y, jvp_pullback
end

function lower_derivative(
        ::EnzymeForwardDerivative, inner, ctx::LoweringContext, shift, direction, order
    )
    directions = fill(direction, order)
    inner = unwrap(inner)
    while iscall(inner) && operation(inner) isa Differential
        D = operation(inner)
        D.order isa Integer && haskey(ctx.iv_global, unwrap(D.x)) ||
            throw(ArgumentError("Unsupported derivative `$(D)` for EnzymeForwardDerivative."))
        append!(directions, fill(ctx.iv_global[unwrap(D.x)], D.order))
        inner = unwrap(only(arguments(inner)))
    end
    1 <= length(directions) <= 4 || throw(
        ArgumentError("EnzymeForwardDerivative supports total derivative orders one through four.")
    )
    iscall(inner) && haskey(ctx.networks, operation(inner)) || throw(
        ArgumentError("EnzymeForwardDerivative requires derivatives of dependent-variable calls.")
    )
    net = ctx.networks[operation(inner)]
    args = unwrap.(arguments(inner))
    all(a -> _isnumber(a) || haskey(ctx.iv_global, a), args) || throw(
        ArgumentError("EnzymeForwardDerivative requires independent-variable or literal network arguments.")
    )
    rows = map(Tuple(directions)) do d
        Tuple(
            j for (j, a) in enumerate(args) if
                get(ctx.iv_global, _isnumber(a) ? unwrap(net.args[j]) : a, 0) == d
        )
    end
    return lower_depvar(inner, ctx, shift, Val(rows))
end
