# Lowering of `Symbolics.Integral` terms into the registered `quadrature` operation.
# For every integral the integrand, the bound expressions and the infinite-bound
# transformations are compiled once into a `QuadratureIntegrand` stored on a
# non-tunable parameter, so the symbolic term is a single static call over the batch
# of collocation points.

# PDEBase derives the independent variables of a system from the arguments of
# dependent-variable calls, so an integrating variable like `τ` in
# `Integral(τ in [0, t])(x(τ))` would demand a domain. Declare an infinite one for
# every integrating variable missing one; it is never sampled because integrating
# variables are bound, not free.
function SciMLBase.symbolic_discretize(
        pdesys::PDESystem, disc::PhysicsInformedNN; checks = true
    )
    _declare_integral_variables(pdesys)
    return invoke(
        SciMLBase.symbolic_discretize,
        Tuple{PDESystem, PDEBase.AbstractOptimizationSystemDiscretization},
        pdesys, disc; checks
    )
end

# Collect the symbols that need a domain entry for `PDEBase.VariableMap`: the
# integrating variables of every `Integral`, plus the non-constant arguments of
# dependent-variable calls inside integrands (a call like `x(t - τ)` registers
# `t - τ` as an independent variable otherwise).
function _collect_integral_vars!(found, ex, dvops, inside = false)
    ex = unwrap(ex)
    iscall(ex) || return found
    op = operation(ex)
    if op isa Symbolics.Integral
        for v in _integral_variables(op.domain)
            any(x -> isequal(x, v), found) || push!(found, v)
        end
        lbs, ubs = _integral_bounds(op.domain.domain)
        for b in vcat(lbs, ubs)
            _collect_integral_vars!(found, b, dvops, inside)
        end
        inside = true
    elseif inside && any(d -> isequal(d, op), dvops)
        for a in arguments(ex)
            a = unwrap(a)
            SymbolicUtils.unwrap_const(a) isa Number && continue
            any(x -> isequal(x, a), found) || push!(found, a)
        end
    end
    for a in arguments(ex)
        _collect_integral_vars!(found, a, dvops, inside)
    end
    return found
end

function _declare_integral_variables(pdesys)
    dom = get_domain(pdesys)
    dom isa AbstractVector || return
    have = Any[]
    for d in dom
        vars = unwrap(d.variables)
        if iscall(vars) && operation(vars) === tuple
            append!(have, unwrap.(arguments(vars)))
        else
            push!(have, vars)
        end
    end
    dvops = Any[operation(unwrap(dv)) for dv in get_dvs(pdesys)]
    for eqs in (get_eqs(pdesys), get_bcs(pdesys)), eq in eqs,
        side in (eq isa Equation ? (eq.lhs, eq.rhs) : (eq,))
        for v in _collect_integral_vars!(Any[], unwrap(side), dvops)
            any(x -> isequal(x, v), have) && continue
            push!(dom, wrap(v) ∈ DomainSets.Interval(-Inf, Inf))
            push!(have, v)
        end
    end
    return nothing
end

"""
    QuadratureIntegrand

Callable stored as the `f` argument of [`quadrature`](@ref). Built by `lower_integral`,
it carries the runtime-compiled evaluators of the integrand and of the original
integration bounds, the per-dimension `σ → τ` transformations used for infinite bounds
with their Jacobians, and the `σ`-space bounds (constants after transformation).

# Fields

* `integrand`: evaluator `(X̃, θcat) -> scalar or 1 × nM` of the integrand, where `X̃`
  is the outer collocation matrix repeated `M` times with the `q` inner integration
  variables appended as extra rows.
* `lbs`, `ubs`: per-dimension evaluators `(Xp, θcat) -> scalar or 1 × nM` of the
  original lower and upper bounds. The bound of the `i`-th integrating variable may
  only depend on the outer variables and the variables `1:i-1` (intervals compose
  lexicographically), so it is evaluated on the partial input `Xp`.
* `σlb`, `σub`: per-dimension lower/upper bounds in the `σ` domain: a constant for
  transformed (infinite) bounds, `nothing` to use the evaluated original bound.
* `Ts`, `dTs`: per-dimension `σ → τ` maps and their `dτ/dσ` Jacobians, broadcast
  elementwise over `(σ, lb, ub)` rows.
"""
struct QuadratureIntegrand
    integrand::Any
    lbs::Vector{Any}
    ubs::Vector{Any}
    σlb::Vector{Any}
    σub::Vector{Any}
    Ts::Vector{Any}
    dTs::Vector{Any}
end

function quadrature(f, X, θ, ξ, w)
    f isa QuadratureIntegrand || throw(
        ArgumentError(
            "`quadrature` expects a `QuadratureIntegrand` as its first argument, \
            got `$(f)`."
        )
    )
    n = size(X, 2)
    M = size(ξ, 2)
    q = size(ξ, 1)
    nM = n * M
    T = promote_type(eltype(X), eltype(ξ))
    # (j-1)*M + k indexes collocation point j at node k; built with broadcast and
    # reshape only, so every AD backend can trace it.
    Xr = reshape(reshape(X, size(X, 1), 1, n) .* ones(T, 1, M, 1), size(X, 1), nM)
    rows = ()
    measure = nothing
    for i in 1:q
        Xp = isempty(rows) ? Xr : vcat(Xr, rows...)
        lb = _quad_row(f.lbs[i](Xp, θ), nM)
        ub = _quad_row(f.ubs[i](Xp, θ), nM)
        σl = f.σlb[i] === nothing ? lb : f.σlb[i]
        σu = f.σub[i] === nothing ? ub : f.σub[i]
        h = (σu .- σl) ./ 2
        c = (σu .+ σl) ./ 2
        σ = c .+ h .* reshape(vec(ξ[i, :] * ones(T, 1, n)), 1, nM)
        m = h .* f.dTs[i].(σ, lb, ub)
        measure = measure === nothing ? m : measure .* m
        rows = (rows..., f.Ts[i].(σ, lb, ub))
    end
    X̃ = isempty(rows) ? Xr : vcat(Xr, rows...)
    F = _quad_row(f.integrand(X̃, θ), nM)
    return reshape(reshape(vec(measure .* F), M, n)' * w, 1, n)
end

_quad_row(r, nM) = r isa AbstractMatrix ? r : fill(r, 1, nM)

struct QuadratureConst{V}
    v::V
end
(e::QuadratureConst)(X, θ) = e.v

"""
    QuadratureEval

`(X̃, θcat) -> value` evaluator wrapping a `Symbolics.build_function` runtime function
whose signature is `(X̃, slots..., consts...)`. `ranges`/`isvec` describe how the flat
`θcat` vector is sliced into the network parameter vectors and scalar `PDESystem`
parameters the expression needs.
"""
struct QuadratureEval{F, C, R, S}
    rgf::F
    consts::C
    ranges::R
    isvec::S
end

function (e::QuadratureEval)(X̃, θcat)
    args = ntuple(length(e.ranges)) do k
        r = e.ranges[k]
        e.isvec[k] ? θcat[r] : θcat[first(r)]
    end
    return e.rgf(X̃, args..., e.consts...)
end

function _integral_variables(domain)
    vars = domain.variables
    if iscall(vars) && operation(vars) === tuple
        return Any[unwrap(v) for v in arguments(vars)]
    end
    return Any[unwrap(vars)]
end

function _interval_endpoints(d)
    return if hasproperty(d, :left) && hasproperty(d, :right)
        d.left, d.right
    else
        DomainSets.infimum(d), DomainSets.supremum(d)
    end
end

function _integral_bounds(domain)
    if domain isa DomainSets.AbstractInterval
        lb, ub = _interval_endpoints(domain)
        return Any[lb], Any[ub]
    elseif domain isa DomainSets.ProductDomain
        lbs = Any[]
        ubs = Any[]
        for c in DomainSets.components(domain)
            lb, ub = _interval_endpoints(c)
            push!(lbs, lb)
            push!(ubs, ub)
        end
        return lbs, ubs
    end
    throw(
        ArgumentError(
            "Unsupported `Integral` domain `$(domain)`; only intervals and products \
            of intervals are supported."
        )
    )
end

function _collect_subexpressions!(found, ex)
    ex = unwrap(ex)
    push!(found, ex)
    iscall(ex) || return found
    for a in arguments(ex)
        _collect_subexpressions!(found, a)
    end
    return found
end

# A bound of the `i`-th integrating variable may depend on the outer variables and on
# the variables `1:i-1`, but not on dependent variables, differentials or nested
# integrals, and not on its own or a later integrating variable (unless that symbol is
# also an outer independent variable, in which case the outer variable is meant).
function _check_integral_bounds(lbs, ubs, ivars, ctx, ex)
    for i in eachindex(ivars), b in (lbs[i], ubs[i])
        for s in _collect_subexpressions!(Any[], b)
            if iscall(s)
                op = operation(s)
                if haskey(ctx.networks, op) || op isa Differential ||
                        op isa Symbolics.Integral
                    throw(
                        ArgumentError(
                            "Invalid expression `$(s)` in the bound `$(b)` of `$(ex)`; \
                            bounds may not contain dependent variables, derivatives \
                            or integrals."
                        )
                    )
                end
            else
                j = findfirst(v -> isequal(v, s), ivars)
                if j !== nothing && j >= i && !haskey(ctx.iv_index, s)
                    throw(
                        ArgumentError(
                            "Bound `$(b)` of `$(ex)` depends on integrating variable \
                            `$(s)`, which is not bound yet at position $(i)."
                        )
                    )
                end
            end
        end
    end
    return nothing
end

function _isinfval(b, v)
    b = unwrap(b)
    b isa Real && return b == v
    return b isa SymbolicUtils.BasicSymbolic && SymbolicUtils.isconst(b) &&
        SymbolicUtils.unwrap_const(b) == v
end

# Map infinite bounds to a finite σ domain: σ ∈ (-1+ϵ, 1-ϵ) for (-∞, ∞),
# σ ∈ [0, 1-ϵ) for (l, ∞) and σ ∈ (-1+ϵ, 0] for (-∞, u). The original bounds are
# still evaluated and passed to the transformations, so symbolic bounds work
# elementwise over the batch.
function _transform_inf_bounds(lbs, ubs)
    ϵ = 1 / 20
    q = length(lbs)
    σlb = Any[nothing for _ in 1:q]
    σub = Any[nothing for _ in 1:q]
    Ts = Any[(σ, l, u) -> σ for _ in 1:q]
    dTs = Any[(σ, l, u) -> one(σ) for _ in 1:q]
    for i in 1:q
        l_inf = _isinfval(lbs[i], -Inf)
        u_inf = _isinfval(ubs[i], Inf)
        if l_inf && u_inf
            σlb[i] = -1 + ϵ
            σub[i] = 1 - ϵ
            Ts[i] = (σ, l, u) -> σ / (1 - σ^2)
            dTs[i] = (σ, l, u) -> (1 + σ^2) / (1 - σ^2)^2
        elseif u_inf
            σlb[i] = 0
            σub[i] = 1 - ϵ
            Ts[i] = (σ, l, u) -> l + σ / (1 - σ)
            dTs[i] = (σ, l, u) -> 1 / (1 - σ)^2
        elseif l_inf
            σlb[i] = -1 + ϵ
            σub[i] = 0
            Ts[i] = (σ, l, u) -> u + σ / (1 + σ)
            dTs[i] = (σ, l, u) -> 1 / (1 + σ)^2
        end
    end
    return σlb, σub, Ts, dTs
end

function _integral_nodes(alg::GaussLegendre, q, T)
    ξ1 = collect(alg.nodes)
    w1 = collect(alg.weights)
    S = alg.subintervals
    if S > 1
        # Bake the composite rule into nodes on [-1, 1] so the affine map applied in
        # `quadrature` stays a single scaling.
        ξ1 = vec([(2k - 1 - S) / S .+ ξ1 ./ S for k in 1:S])
        w1 = vec(fill(w1 ./ S, S))
    end
    ξ = _product_matrix(fill(ξ1, q))
    w = vec(prod(_product_matrix(fill(w1, q)); dims = 1))
    return T.(ξ), T.(w)
end
function _integral_nodes(alg, q, T)
    throw(
        ArgumentError(
            "`integral_alg` must be a fixed-node rule; only `Integrals.GaussLegendre` \
            is supported, got `$(alg)`."
        )
    )
end

# Strip indexing (`xs_int[1:1, :]`) to the underlying symbolic.
function _rootvar(s)
    s = unwrap(s)
    while iscall(s) && operation(s) === getindex
        s = unwrap(arguments(s)[1])
    end
    return s
end

# The collocation matrix translated by the finite-difference `shift`: differentiating
# an integral term translates the whole evaluation, bounds included.
function _shifted_xs(ctx::LoweringContext, shift)
    T = ctx.eltype
    ctx.xs === nothing && return zeros(T, 0, 1)
    X = wrap(ctx.xs)
    s = zeros(T, maximum(values(ctx.iv_index)))
    for (v, row) in ctx.iv_index
        s[row] = shift[ctx.iv_global[v]]
    end
    iszero(s) && return X
    return X .+ reshape(s, :, 1)
end

function _theta_cat(slots, isvec, T)
    isempty(slots) && return wrap(T[])
    length(slots) == 1 &&
        return isvec[1] ? wrap(slots[1]) : Num[wrap(slots[1])]
    parts = map(eachindex(slots)) do i
        isvec[i] ? wrap(slots[i]) : Num[wrap(slots[i])]
    end
    return foldl(nn_veccat, parts)
end

function _const_value(s)
    v = getdefault(s)
    v === nothing && throw(
        ArgumentError("Symbolic `$(s)` inside an `Integral` has no default value.")
    )
    return v
end

# Compile the integrand and bound expressions into one `QuadratureIntegrand` and the
# matching `θcat` expression: the runtime functions take `(X̃, slots..., consts...)`
# where `slots` are the network parameters and `PDESystem` parameters the lowered
# expressions use, and `consts` everything else (network callables, nested quadrature
# parameters).
function _quadrature_integrand(F, lbE, ubE, σlb, σub, Ts, dTs, xs_int, ctx)
    needed = Any[]
    for e in (F, lbE..., ubE...), s in Symbolics.get_variables(unwrap(e))
        r = _rootvar(s)
        (isequal(r, xs_int) || SymbolicUtils.isconst(r)) && continue
        any(x -> isequal(x, r), needed) || push!(needed, r)
    end
    slots = Any[]
    isvec = Bool[]
    consts = Any[]
    for r in needed
        if any(net -> isequal(unwrap(net.θ), r), values(ctx.networks))
            push!(slots, r)
            push!(isvec, true)
        elseif any(p -> isequal(unwrap(p), r), values(ctx.params))
            push!(slots, r)
            push!(isvec, false)
        else
            push!(consts, r)
        end
    end
    lens = [isvec[i] ? length(unwrap(slots[i])) : 1 for i in eachindex(slots)]
    offs = cumsum([1; lens])
    ranges = [offs[i]:(offs[i + 1] - 1) for i in eachindex(lens)]
    constvals = Tuple(_const_value(c) for c in consts)
    mk_eval = function (e)
        e = unwrap(e)
        _isnumber(e) && return QuadratureConst(_number(e))
        rgf = Symbolics.build_function(
            e, wrap(xs_int), wrap.(slots)..., wrap.(consts)...; expression = Val{false}
        )[1]
        return QuadratureEval(rgf, constvals, ranges, isvec)
    end
    ev = QuadratureIntegrand(
        mk_eval(F), Any[mk_eval(e) for e in lbE],
        Any[mk_eval(e) for e in ubE], σlb, σub, Ts, dTs
    )
    return ev, _theta_cat(slots, isvec, ctx.eltype)
end

function lower_integral(ex, ctx::LoweringContext, shift)
    T = ctx.eltype
    Iop = operation(ex)
    integrand = unwrap(only(arguments(ex)))
    ivars = _integral_variables(Iop.domain)
    q = length(ivars)
    lbs, ubs = _integral_bounds(Iop.domain.domain)
    length(lbs) == q || throw(
        ArgumentError("Domain of `$(Iop)` does not match its integrating variables.")
    )
    _check_integral_bounds(lbs, ubs, ivars, ctx, ex)
    σlb, σub, Ts, dTs = _transform_inf_bounds(lbs, ubs)
    ξ, w = _integral_nodes(ctx.integral_alg, q, T)
    M = size(ξ, 2)
    nM = ctx.npoints * M
    # `iv_index` values are rows of `ctx.xs`; an integrating variable that shadows an
    # outer variable name replaces its entry, so the row count is the largest value.
    d = ctx.xs === nothing ? 0 : maximum(values(ctx.iv_index))
    nivs = maximum(values(ctx.iv_global))
    k = div(length(ctx.extras), 3) + 1
    xs_int = unwrap(only(@parameters $(Symbol(:xsi_, ctx.tag, :_, k))[1:(d + q), 1:nM]))
    # The integrating variables are appended to the network input as extra rows. They
    # shadow outer variables of the same name inside the integrand.
    iv_index_i = copy(ctx.iv_index)
    iv_global_i = copy(ctx.iv_global)
    for (j, v) in enumerate(ivars)
        iv_index_i[v] = d + j
        iv_global_i[v] = nivs + j
    end
    ctx_inner = LoweringContext(
        xs_int, iv_index_i, iv_global_i, ctx.networks, ctx.params,
        ctx.derivative, nM, T, ctx.integral_alg, ctx.extras, ctx.tag
    )
    shift_inner = zeros(T, nivs + q)
    F = lower(integrand, ctx_inner, shift_inner)
    lbE = Any[]
    ubE = Any[]
    for i in 1:q
        iv_index_b = copy(ctx.iv_index)
        iv_global_b = copy(ctx.iv_global)
        for j in 1:(i - 1)
            iv_index_b[ivars[j]] = d + j
            iv_global_b[ivars[j]] = nivs + j
        end
        ctx_b = LoweringContext(
            xs_int, iv_index_b, iv_global_b, ctx.networks, ctx.params,
            ctx.derivative, nM, T, ctx.integral_alg, ctx.extras, ctx.tag
        )
        push!(lbE, lower(lbs[i], ctx_b, shift_inner))
        push!(ubE, lower(ubs[i], ctx_b, shift_inner))
    end
    ev, θcat = _quadrature_integrand(F, lbE, ubE, σlb, σub, Ts, dTs, xs_int, ctx)
    # Recompute the index: nested integrals in `F` have already pushed their extras.
    k = div(length(ctx.extras), 3) + 1
    qf = only(@parameters ($(Symbol(:qf_, ctx.tag, :_, k))::typeof(ev))(..) = ev [tunable = false])
    ξs = only(@parameters $(Symbol(:xi_, ctx.tag, :_, k))[1:q, 1:M] = ξ [tunable = false])
    ws = only(@parameters $(Symbol(:wi_, ctx.tag, :_, k))[1:M] = w [tunable = false])
    push!(ctx.extras, qf, ξs, ws)
    return wrap(quadrature(qf, _shifted_xs(ctx, shift), θcat, ξs, ws))
end
