# Array arguments of dependent variables (`u(t, x)` with `@parameters x[1:d]`) are
# packed to the scalar input `[t; vec(x)]` before PDEBase builds a `VariableMap`.
# The map only resolves scalar independent variables. `DepvarPacking` keeps the
# declared grouping so the solution interface can invert the packing.

const _PACKING_TLS = :NeuralPDE_array_argument_packing

struct DepvarPacking
    op::Any
    original::Any
    expanded::Any
    groups::Vector{ArgumentGroup}
end

struct ArrayPacking
    depvars::Vector{DepvarPacking}
    nflat::Int
end

_current_packing() = get(task_local_storage(), _PACKING_TLS, nothing)

function _packing_entry(op)
    packing = _current_packing()
    packing === nothing && return nothing
    for entry in packing.depvars
        isequal(entry.op, op) && return entry
    end
    return nothing
end

function _symbolic_array(a)
    if a isa SymbolicUtils.BasicSymbolic
        return SymbolicUtils.symtype(a) <: AbstractArray ? wrap(a) : nothing
    end
    a isa AbstractArray || return nothing
    au = unwrap(a)
    if au isa SymbolicUtils.BasicSymbolic && SymbolicUtils.symtype(au) <: AbstractArray
        return a
    end
    return nothing
end

function _as_scalar(c)
    cu = unwrap(c)
    if cu isa Number
        return cu
    elseif cu isa SymbolicUtils.BasicSymbolic && SymbolicUtils.isconst(cu)
        return SymbolicUtils.unwrap_const(cu)
    else
        return c
    end
end

function _fixed_shape(au)
    sz = try
        size(au)
    catch
        return nothing
    end
    sz isa Tuple && !isempty(sz) && all(n -> n isa Integer && n > 0, sz) || return nothing
    return sz
end

function _component_list(arr)
    au = unwrap(arr)
    # `CartesianIndices` walks column-major order, which is `vec`. Symbolic matrices
    # reject linear `getindex`; a 1-d `array_literal` still simplifies under `(i,)`.
    sz = _fixed_shape(au)
    sz === nothing && throw(
        ArgumentError(
            "Array argument `$arr` does not have a fixed shape. Declare independent \
            variables with explicit indices, for example `@parameters x[1:d]`."
        )
    )
    # A comprehension over `CartesianIndices` would keep the array shape. `vec` order
    # is the iteration order, stored as a flat vector of slots.
    comps = Vector{Any}(undef, prod(sz))
    i = 0
    for I in CartesianIndices(sz)
        i += 1
        comps[i] = _as_scalar(au[Tuple(I)...])
    end
    return comps
end

function _packed_components(a)
    arr = _symbolic_array(a)
    arr !== nothing && return _component_list(arr)
    if a isa AbstractArray && !(a isa SymbolicUtils.BasicSymbolic)
        return Any[vec(collect(a))...]
    end
    return nothing
end

_is_depvar_op(op, ops) = any(o -> isequal(o, op), ops)

function _depvar_ops(pdesys)
    ops = Any[]
    for dv in get_dvs(pdesys)
        dv = unwrap(dv)
        iscall(dv) || continue
        push!(ops, operation(dv))
    end
    return ops
end

function _flat_length(args)
    n = 0
    for a in args
        comps = _packed_components(a)
        n += comps === nothing ? 1 : length(comps)
    end
    return n
end

function _groups_of(dv)
    return map(arguments(unwrap(dv))) do a
        comps = _packed_components(a)
        comps === nothing ? ArgumentGroup(a, Any[a], false) : ArgumentGroup(a, comps, true)
    end
end

function _expanded_call(op, groups)
    args = Any[]
    for g in groups
        append!(args, g.components)
    end
    return op(args...)
end

function _rewrite_ex(ex, ops)
    exu = unwrap(ex)
    iscall(exu) || return ex
    op = operation(exu)
    raw = arguments(exu)
    newargs = map(a -> _rewrite_ex(a, ops), raw)
    if _is_depvar_op(op, ops)
        return op(_expand_call_arguments(newargs, ops)...)
    end
    for i in eachindex(newargs)
        isequal(unwrap(newargs[i]), unwrap(raw[i])) && continue
        return op(newargs...)
    end
    return ex
end

function _expand_call_arguments(args, ops)
    out = Any[]
    for a in args
        comps = _packed_components(a)
        if comps === nothing
            push!(out, a)
        else
            for c in comps
                push!(out, _rewrite_ex(c, ops))
            end
        end
    end
    return out
end

function _rewrite_piece(eq::Equation, ops)
    return _rewrite_ex(eq.lhs, ops) ~ _rewrite_ex(eq.rhs, ops)
end
function _rewrite_piece(eq::Pair, ops)
    return _rewrite_ex(eq.first, ops) => _rewrite_ex(eq.second, ops)
end
_rewrite_piece(eqs::AbstractVector, ops) = map(eq -> _rewrite_piece(eq, ops), eqs)
_rewrite_piece(eq, ::Any) = eq

function _scan_array_args!(found, ex, ops)
    found[] && return
    if ex isa Equation
        _scan_array_args!(found, ex.lhs, ops)
        _scan_array_args!(found, ex.rhs, ops)
        return nothing
    elseif ex isa Pair
        _scan_array_args!(found, ex.first, ops)
        _scan_array_args!(found, ex.second, ops)
        return nothing
    end
    ex isa AbstractArray && _symbolic_array(ex) === nothing && return _scan_elements!(found, ex, ops)
    exu = unwrap(ex)
    iscall(exu) || return
    op = operation(exu)
    args = arguments(exu)
    if _is_depvar_op(op, ops)
        for a in args
            _packed_components(a) === nothing || (found[] = true)
        end
    end
    for a in args
        _scan_array_args!(found, a, ops)
    end
    return nothing
end

function _scan_elements!(found, arr, ops)
    for el in arr
        _scan_array_args!(found, el, ops)
    end
    return nothing
end

function _needs_expansion(pdesys, ops)
    for iv in get_ivs(pdesys)
        _symbolic_array(iv) !== nothing && return true
    end
    for d in get_domain(pdesys)
        _symbolic_array(d.variables) !== nothing && return true
    end
    found = Ref(false)
    for piece in (get_eqs(pdesys)..., get_bcs(pdesys)..., get_dvs(pdesys)...)
        _scan_array_args!(found, piece, ops)
        found[] && return true
    end
    return false
end

function _walk_calls!(f, ex, ops)
    if ex isa Equation
        _walk_calls!(f, ex.lhs, ops)
        _walk_calls!(f, ex.rhs, ops)
        return nothing
    elseif ex isa Pair
        _walk_calls!(f, ex.first, ops)
        _walk_calls!(f, ex.second, ops)
        return nothing
    elseif ex isa AbstractArray && _symbolic_array(ex) === nothing
        for el in ex
            _walk_calls!(f, el, ops)
        end
        return nothing
    end
    exu = unwrap(ex)
    iscall(exu) || return nothing
    op = operation(exu)
    if _is_depvar_op(op, ops)
        f(exu)
    end
    for a in arguments(exu)
        _walk_calls!(f, a, ops)
    end
    return nothing
end

function _declared_flat_length(op, pdesys)
    for dv in get_dvs(pdesys)
        dvu = unwrap(dv)
        iscall(dvu) || continue
        isequal(operation(dvu), op) || continue
        return _flat_length(arguments(dvu))
    end
    return nothing
end

function _check_signature_lengths(pdesys, ops)
    check = function (ex)
        op = operation(ex)
        expected = _declared_flat_length(op, pdesys)
        expected === nothing && return nothing
        n = _flat_length(arguments(ex))
        n == expected || throw(
            ArgumentError(
                "Call `$ex` packs to $n network inputs; the declared signature of `$op` \
                packs to $expected."
            )
        )
        return nothing
    end
    for piece in (get_eqs(pdesys)..., get_bcs(pdesys)..., get_dvs(pdesys)...)
        _walk_calls!(check, piece, ops)
    end
    return nothing
end

_contains_symbol(list, sym) = any(v -> isequal(unwrap(v), unwrap(sym)), list)

function _expand_ivs(ivs)
    flat = Any[]
    parents = Tuple{Any, Any, Int}[]
    for iv in ivs
        arr = _symbolic_array(iv)
        if arr === nothing
            _contains_symbol(flat, iv) && throw(
                ArgumentError("Independent variable `$iv` is listed more than once.")
            )
            push!(flat, iv)
            continue
        end
        for (k, c) in enumerate(_component_list(arr))
            _contains_symbol(flat, c) && throw(
                ArgumentError(
                    "Component `$c` of `$iv` is already an independent variable. List the \
                    array or its components, not both."
                )
            )
            push!(flat, c)
            push!(parents, (unwrap(c), arr, k))
        end
    end
    return flat, parents
end

function _append_missing_components!(flat, parents, entries)
    for entry in entries
        for g in entry.groups
            g.array || continue
            for (k, c) in enumerate(g.components)
                _contains_symbol(flat, c) && continue
                push!(flat, c)
                push!(parents, (unwrap(c), g.symbol, k))
            end
        end
    end
    return flat
end

function _split_domain_entries(domains)
    scalar = Any[]
    arrays = Any[]
    for d in domains
        if _symbolic_array(d.variables) !== nothing
            push!(arrays, d)
        else
            push!(scalar, d)
        end
    end
    return scalar, arrays
end

function _variables_match(vars, comp)
    u = unwrap(comp)
    if (vars isa Tuple || vars isa AbstractArray) && _symbolic_array(vars) === nothing &&
            !(vars isa SymbolicUtils.BasicSymbolic)
        return any(v -> isequal(unwrap(v), u), vars)
    end
    return isequal(unwrap(vars), u)
end

function _scalar_domain(comp, scalar_domains)
    for d in scalar_domains
        _variables_match(d.variables, comp) && return d.domain
    end
    return nothing
end

function _parent_component(comp, parents)
    u = unwrap(comp)
    for (cu, arr, idx) in parents
        isequal(cu, u) && return (arr, idx)
    end
    return nothing
end

function _product_factor(arr, index, array_domains)
    uarr = unwrap(arr)
    for d in array_domains
        isequal(unwrap(d.variables), uarr) || continue
        dom = d.domain
        dom isa DomainSets.ProductDomain || throw(
            ArgumentError(
                "Domain of array independent variable `$arr` must be a product domain \
                (one factor per component, in `vec` order) or one interval per component."
            )
        )
        factors = collect(DomainSets.factors(dom))
        n = length(_component_list(arr))
        length(factors) == n || throw(
            ArgumentError(
                "Product domain of `$arr` has $(length(factors)) factors; `$arr` has \
                $n components in `vec` order."
            )
        )
        return factors[index]
    end
    return nothing
end

function _domain_for(comp, parents, scalar_domains, array_domains)
    dom = _scalar_domain(comp, scalar_domains)
    dom !== nothing && return comp ∈ dom
    parent = _parent_component(comp, parents)
    parent === nothing && throw(
        ArgumentError(
            "No domain for independent variable `$comp`. Give `$comp ∈ Interval(a, b)`, \
            or a product domain for the array it belongs to."
        )
    )
    arr, idx = parent
    factor = _product_factor(arr, idx, array_domains)
    factor === nothing && throw(
        ArgumentError(
            "No domain for `$comp`. Give a per-component interval, or a product domain \
            for `$arr`."
        )
    )
    return comp ∈ factor
end

function _packing_entries(pdesys)
    entries = DepvarPacking[]
    for dv in get_dvs(pdesys)
        dvu = unwrap(dv)
        iscall(dvu) || continue
        groups = _groups_of(dvu)
        any(g -> g.array, groups) || continue
        op = operation(dvu)
        push!(entries, DepvarPacking(op, dvu, _expanded_call(op, groups), groups))
    end
    return entries
end

function _rebuild_pdesys(pdesys, eqs, bcs, domain, ivs, dvs, analytic)
    af = analytic === nothing ? getfield(pdesys, :analytic_func) : nothing
    return PDESystem(
        eqs, bcs, domain, ivs, dvs, get_ps(pdesys);
        initial_conditions = getfield(pdesys, :initial_conditions),
        systems = getfield(pdesys, :systems),
        connector_type = getfield(pdesys, :connector_type),
        metadata = getfield(pdesys, :metadata),
        analytic = analytic,
        analytic_func = af,
        gui_metadata = getfield(pdesys, :gui_metadata),
        description = getfield(pdesys, :description),
        name = getfield(pdesys, :name),
        checks = false
    )
end

"""
    expand_array_arguments(pdesys) -> (pdesys, packing)

Return `pdesys` unchanged and `packing = nothing` when every dependent-variable
argument is scalar. Otherwise return a system whose dependent-variable calls and
independent-variable list use the packed scalar components, plus the `ArrayPacking`
that records the declared grouping.
"""
function expand_array_arguments(pdesys::PDESystem)
    ops = _depvar_ops(pdesys)
    _needs_expansion(pdesys, ops) || return pdesys, nothing
    _check_signature_lengths(pdesys, ops)
    entries = _packing_entries(pdesys)
    flat, parents = _expand_ivs(get_ivs(pdesys))
    _append_missing_components!(flat, parents, entries)
    scalar_domains, array_domains = _split_domain_entries(get_domain(pdesys))
    domains = map(flat) do comp
        _domain_for(comp, parents, scalar_domains, array_domains)
    end
    eqs = _rewrite_piece(get_eqs(pdesys), ops)
    bcs = _rewrite_piece(get_bcs(pdesys), ops)
    dvs = map(dv -> _rewrite_ex(dv, ops), get_dvs(pdesys))
    analytic = getfield(pdesys, :analytic)
    analytic === nothing || (analytic = _rewrite_piece(analytic, ops))
    expanded = _rebuild_pdesys(pdesys, eqs, bcs, domains, flat, dvs, analytic)
    return expanded, ArrayPacking(entries, length(flat))
end

function _check_grid_spacing(disc, packing)
    packing === nothing && return nothing
    disc.strategy isa GridTraining || return nothing
    dx = disc.strategy.dx
    dx isa Number && return nothing
    nflat = packing.nflat
    length(dx) == nflat || throw(
        ArgumentError(
            "`GridTraining` spacing has length $(length(dx)); the packed independent \
            variables have length $nflat. Array arguments are expanded in `vec` order."
        )
    )
    return nothing
end

function _argument_shape(g::ArgumentGroup)
    sz = _fixed_shape(unwrap(g.symbol))
    return sz === nothing ? (length(g.components),) : sz
end

function _points_for(g::ArgumentGroup, a, sol)
    w = length(g.components)
    if !g.array
        a isa Colon && return collect(_iv_grid(sol, only(g.components)))
        return a isa Number ? [a] : collect(a)
    end
    a isa Colon && throw(
        ArgumentError(
            "`:` stands for the evaluation grid of a scalar argument. Index \
            `sol[$(g.symbol)]` for the tensor-product grid of an array argument."
        )
    )
    # A numeric array of the declared shape is one point, in column-major `vec` order.
    # A length-`w` vector is that same point when the caller has already flattened it.
    if a isa AbstractArray{<:Number} && size(a) == _argument_shape(g)
        return Any[vec(a)]
    end
    if a isa AbstractVector{<:Number}
        length(a) == w || throw(
            ArgumentError(
                "Array argument `$(g.symbol)` has $w components; got a vector of length \
                $(length(a))."
            )
        )
        return Any[collect(a)]
    end
    if a isa AbstractMatrix{<:Number}
        if size(a, 1) == w
            return Any[collect(@view a[:, j]) for j in axes(a, 2)]
        elseif size(a, 2) == w && size(a, 1) != w
            return Any[collect(@view a[i, :]) for i in axes(a, 1)]
        else
            throw(
                ArgumentError(
                    "Array argument `$(g.symbol)` has $w components. Pass a length-$w \
                    vector, or a matrix with $w rows (one point per column)."
                )
            )
        end
    end
    shape = _argument_shape(g)
    if a isa AbstractVector && all(p -> p isa AbstractArray{<:Number}, a)
        for p in a
            size(p) == shape || (p isa AbstractVector{<:Number} && length(p) == w) || throw(
                ArgumentError(
                    "Array argument `$(g.symbol)` has shape $shape; got a point of \
                    size $(size(p))."
                )
            )
        end
        return Any[vec(p) for p in a]
    end
    throw(
        ArgumentError(
            "Array argument `$(g.symbol)` expects a length-$w vector (one point) or a \
            matrix of such points, got `$(typeof(a))`."
        )
    )
end

function _iv_grid(sol, sym)
    i = findfirst(v -> isequal(unwrap(v), unwrap(sym)), sol.ivs)
    i === nothing && throw(ArgumentError("Independent variable `$sym` is not in the solution."))
    return sol.ivdomain[i]
end

_is_grouped_call(groups, args) =
    groups !== nothing && length(args) == length(groups) && any(g -> g.array, groups)

function _flatten_point(pts, groups, ::Type{T}) where {T}
    col = T[]
    for (p, g) in zip(pts, groups)
        if g.array
            append!(col, T.(p))
        else
            push!(col, T(p))
        end
    end
    return col
end

function _eval_grouped(f, groups, args, sol)
    length(args) == length(groups) || throw(
        ArgumentError(
            "Expected $(length(groups)) arguments for the declared call, got $(length(args))."
        )
    )
    pointsets = map((g, a) -> _points_for(g, a, sol), groups, args)
    dims = length.(pointsets)
    n = prod(dims)
    T = Float64
    for pts in pointsets, p in pts
        T = promote_type(T, p isa Number ? typeof(p) : eltype(p))
    end
    T = float(T)
    d_in = sum(g -> length(g.components), groups; init = 0)
    X = Matrix{T}(undef, d_in, n)
    for (j, pts) in enumerate(Iterators.product(pointsets...))
        X[:, j] = _flatten_point(pts, groups, T)
    end
    vals = vec(f(X))
    # A single point (a number, or one vector for an array argument) does not add an
    # axis, matching `sol(t, xvec)` for one `xvec`.
    keep = [i for i in eachindex(dims) if dims[i] != 1]
    if isempty(keep)
        return only(vals)
    elseif length(keep) == 1
        return vec(vals)
    else
        return reshape(vals, dims[keep]...)
    end
end
