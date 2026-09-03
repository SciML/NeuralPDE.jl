# Symbolic Parser Pipeline for Physics-Informed Neural Networks (PINNs)
# Batched, Finite-Difference Vectorized Residual Kernel Generator

using Symbolics: Symbolics, Differential, unwrap, expand_derivatives, @register_symbolic
using SymbolicUtils: SymbolicUtils
using ModelingToolkit: toexpr
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
using ComponentArrays: ComponentArray
using Integrals: Integrals

# ----------------------------------------------------------------------
# 1. Batched Runtime Operators & Step Sizes
# ----------------------------------------------------------------------

"""
    get_fd_step(::Type{T}, order::Int)

Calculate the optimal finite-difference step size for derivative of given order and precision.
"""
@inline function get_fd_step(::Type{T}, order::Int) where {T}
    return eps(T)^(one(T) / (T(2) + T(order)))
end
@inline get_fd_step(x::AbstractArray{T}, order::Int) where {T} = get_fd_step(T, order)

"""
    get_pinn_theta(θ, dv_idx::Int, is_multioutput::Bool)

Extract parameters for the `dv_idx`-th dependent variable neural network.
"""
function get_pinn_theta(θ, dv_idx::Int, is_multioutput::Bool)
    if !is_multioutput
        if θ isa ComponentArray && haskey(θ, :depvar)
            return θ.depvar
        elseif θ isa NamedTuple && haskey(θ, :depvar)
            return θ.depvar
        elseif hasproperty(θ, :depvar)
            return getproperty(θ, :depvar)
        else
            return θ
        end
    end
    if θ isa ComponentArray
        if haskey(θ, :depvar)
            dep = θ.depvar
            if dep isa ComponentArray
                pnames = propertynames(dep)
                return getproperty(dep, pnames[dv_idx])
            elseif dep isa NamedTuple || dep isa Tuple
                return dep[dv_idx]
            else
                return dep
            end
        else
            pnames = propertynames(θ)
            return getproperty(θ, pnames[dv_idx])
        end
    elseif θ isa NamedTuple
        if haskey(θ, :depvar)
            return θ.depvar[dv_idx]
        else
            return θ[dv_idx]
        end
    elseif θ isa AbstractVector{<:AbstractVector} || θ isa Tuple
        return θ[dv_idx]
    else
        return θ
    end
end

"""
    pde_param_value(θ, idx, default_val)

Return the `idx`-th PDE parameter from the optimization parameter container when
parameter estimation is enabled, otherwise use the numeric value supplied by the
`PDESystem` defaults.
"""
function pde_param_value(
        θ::Union{ComponentArray, NamedTuple, AbstractArray, Tuple}, idx::Int,
        default_val::Float64
    )
    if θ isa ComponentArray && haskey(θ, :p)
        return θ.p[idx]
    elseif θ isa NamedTuple && haskey(θ, :p)
        return θ.p[idx]
    elseif hasproperty(θ, :p)
        return getproperty(θ, :p)[idx]
    elseif isfinite(default_val)
        return default_val
    end

    throw(ArgumentError("PDE parameter $idx has no runtime value. Provide an initial condition/default or enable parameter estimation."))
end

"""
    cord_slice(cord, row_idx::Int)

Extract single coordinate row from `cord` matrix as a (1, N) matrix.
"""
cord_slice(cord::AbstractMatrix, row_idx::Int) = cord[[row_idx], :]

"""
    cord_slice_rows(cord, row_indices::Vector{Int})

Extract multiple coordinate rows from `cord` matrix as a (D_sub, N) matrix.
"""
cord_slice_rows(cord::AbstractMatrix, row_indices::Vector{Int}) = cord[row_indices, :]

"""
    phi_eval(phi, cord_batch, θ, dv_idx, is_multioutput)

Evaluate trial solution / neural network over batched coordinate matrix (D, N) -> (1, N).
"""
function phi_eval(phi, cord_batch::AbstractMatrix, θ, dv_idx::Int, is_multioutput::Bool)
    θ_dv = get_pinn_theta(θ, dv_idx, is_multioutput)
    phi_dv = (is_multioutput && (phi isa Tuple || phi isa AbstractVector)) ? phi[dv_idx] : phi
    return phi_dv(cord_batch, θ_dv)
end

"""
    deriv_fd(derivative, phi, cord_batch, θ, directions, dv_idx, is_multioutput)

Batched central finite difference derivative operator over all collocation columns simultaneously.
"""
function deriv_fd(derivative, phi, cord_batch::AbstractMatrix{T}, θ, directions::Vector{Int}, dv_idx::Int, is_multioutput::Bool) where {T}
    if isempty(directions)
        return phi_eval(phi, cord_batch, θ, dv_idx, is_multioutput)
    end

    order = length(directions)
    D = size(cord_batch, 1)
    θ_dv = get_pinn_theta(θ, dv_idx, is_multioutput)
    phi_dv = (is_multioutput && (phi isa Tuple || phi isa AbstractVector)) ? phi[dv_idx] : phi
    u_fn = (c, p, net) -> net(c, p)

    step = get_fd_step(T, order)
    εs = @ignore_derivatives begin
        map(directions) do dir
            [ifelse(i == dir, step, zero(T)) for i in 1:D]
        end
    end

    return derivative(phi_dv, u_fn, cord_batch, εs, order, θ_dv)
end

function eval_numeric_integral(integrand_fn, cord_batch::AbstractMatrix{T}, phi, θ, derivative, int_var_indices::Vector{Int}, lb_vals::Vector{Float64}, ub_vals::Vector{Float64}, lb_col_indices::Vector{Int}, ub_col_indices::Vector{Int}, is_inf_vec::Vector{Int}) where {T}
    N = size(cord_batch, 2)
    D = size(cord_batch, 1)
    K = length(int_var_indices)

    sol_array = map(1:N) do j
        col = @ignore_derivatives cord_batch[:, j]

        if K == 1
            idx = int_var_indices[1]
            is_inf = is_inf_vec[1]
            a = (lb_col_indices[1] > 0) ? col[lb_col_indices[1]] : T(lb_vals[1])
            b = (ub_col_indices[1] > 0) ? col[ub_col_indices[1]] : T(ub_vals[1])

            if is_inf == 0 && a >= b
                return zero(T)
            end

            if is_inf == 1
                integrand = (t, p) -> begin
                    s = T(v_semiinf_eval(t, a, true))
                    jac = T(v_semiinf_jacobian(t, true))
                    cord_eval = @ignore_derivatives begin
                        c = copy(col)
                        c[idx] = s
                        reshape(c, D, 1)
                    end
                    return integrand_fn(cord_eval, phi, p, derivative)[1] * jac
                end
                prob = Integrals.IntegralProblem(integrand, (zero(T), one(T) - T(1.0e-4)), θ)
                sol = Integrals.solve(prob, Integrals.QuadGKJL(), reltol = 1.0e-4, abstol = 1.0e-4)
                return sol.u
            elseif is_inf == 2
                integrand = (t, p) -> begin
                    s = T(v_inf_eval(t))
                    jac = T(v_inf_jacobian(t))
                    cord_eval = @ignore_derivatives begin
                        c = copy(col)
                        c[idx] = s
                        reshape(c, D, 1)
                    end
                    return integrand_fn(cord_eval, phi, p, derivative)[1] * jac
                end
                prob = Integrals.IntegralProblem(integrand, (-one(T) + T(1.0e-4), one(T) - T(1.0e-4)), θ)
                sol = Integrals.solve(prob, Integrals.QuadGKJL(), reltol = 1.0e-4, abstol = 1.0e-4)
                return sol.u
            else
                integrand = (x, p) -> begin
                    cord_eval = @ignore_derivatives begin
                        c = copy(col)
                        c[idx] = x
                        reshape(c, D, 1)
                    end
                    return integrand_fn(cord_eval, phi, p, derivative)[1]
                end
                prob = Integrals.IntegralProblem(integrand, (a, b), θ)
                sol = Integrals.solve(prob, Integrals.QuadGKJL(), reltol = 1.0e-4, abstol = 1.0e-4)
                return sol.u
            end
        else
            lb_pt = @ignore_derivatives T[(lb_col_indices[k] > 0) ? col[lb_col_indices[k]] : lb_vals[k] for k in 1:K]
            ub_pt = @ignore_derivatives T[(ub_col_indices[k] > 0) ? col[ub_col_indices[k]] : ub_vals[k] for k in 1:K]

            if any(lb_pt .>= ub_pt)
                return zero(T)
            end

            integrand_nd = (x_vec, p) -> begin
                cord_eval = @ignore_derivatives begin
                    c = copy(col)
                    for k in 1:K
                        c[int_var_indices[k]] = x_vec[k]
                    end
                    reshape(c, D, 1)
                end
                return integrand_fn(cord_eval, phi, p, derivative)[1]
            end
            prob = Integrals.IntegralProblem(integrand_nd, (lb_pt, ub_pt), θ)
            sol = Integrals.solve(prob, Integrals.HCubatureJL(), reltol = 1.0e-3, abstol = 1.0e-3)
            return sol.u
        end
    end

    return reshape(T.(sol_array), 1, N)
end

function construct_bc_cord(cord::AbstractMatrix{T}, row_types::Vector{Int}, row_indices::Vector{Int}, row_consts::Vector{Float64}) where {T}
    D = length(row_types)

    if D == size(cord, 1) && row_types == fill(1, D) && row_indices == collect(1:D)
        return cord
    end

    rows = map(1:D) do k
        if row_types[k] == 1
            idx = row_indices[k]
            cord[idx:idx, :]
        else
            val = T(row_consts[k])
            (cord[1:1, :] .* zero(T)) .+ val
        end
    end

    return reduce(vcat, rows)
end

function ChainRulesCore.rrule(::typeof(cord_slice), cord, row_idx::Int)
    result = cord_slice(cord, row_idx)
    pullback(Δ) = (
        ChainRulesCore.NoTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.NoTangent(),
    )
    return result, pullback
end

function ChainRulesCore.rrule(::typeof(cord_slice_rows), cord, row_indices::Vector{Int})
    result = cord_slice_rows(cord, row_indices)
    pullback(Δ) = (
        ChainRulesCore.NoTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.NoTangent(),
    )
    return result, pullback
end

function ChainRulesCore.rrule(
        ::typeof(construct_bc_cord), cord, row_types::Vector{Int},
        row_indices::Vector{Int}, row_consts::Vector{Float64}
    )
    result = construct_bc_cord(cord, row_types, row_indices, row_consts)
    pullback(Δ) = (
        ChainRulesCore.NoTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
    )
    return result, pullback
end

# Register parser primitives as scalar symbolic placeholders. At runtime several
# of them return batched row matrices, but scalar shape during rewriting lets
# ordinary PDE algebra lower cleanly before `_dot_` broadcasts the generated Expr.
@register_symbolic cord_slice(cord, row_idx)
@register_symbolic cord_slice_rows(cord, row_indices)
@register_symbolic construct_bc_cord(cord, row_types, row_indices, row_consts)
@register_symbolic phi_eval(phi, cord, θ, dv_idx, is_multioutput)
@register_symbolic deriv_fd(derivative, phi, cord, θ, directions, dv_idx, is_multioutput)
# eval_numeric_integral is constructed via SymbolicUtils.term only. Registering
# its long argument list triggers Symbolics precompilation recursion on Julia 1.12.

SymbolicUtils.promote_symtype(::typeof(cord_slice), args...) = Real
SymbolicUtils.promote_symtype(::typeof(cord_slice_rows), args...) = Real
SymbolicUtils.promote_symtype(::typeof(construct_bc_cord), args...) = Real
SymbolicUtils.promote_symtype(::typeof(phi_eval), args...) = Real
SymbolicUtils.promote_symtype(::typeof(deriv_fd), args...) = Real
SymbolicUtils.promote_symtype(::typeof(pde_param_value), args...) = Real
SymbolicUtils.promote_symtype(::typeof(eval_numeric_integral), args...) = Real

function SymbolicUtils.promote_shape(
        ::Union{
            typeof(cord_slice),
            typeof(cord_slice_rows),
            typeof(construct_bc_cord),
            typeof(phi_eval),
            typeof(deriv_fd),
            typeof(pde_param_value),
            typeof(eval_numeric_integral),
        },
        args::SymbolicUtils.ShapeT...
    )
    @nospecialize args
    return SymbolicUtils.ShapeVecT()
end

# ----------------------------------------------------------------------
# 2. Term Rewriter via SymbolicUtils Prewalk
# ----------------------------------------------------------------------

function unwrap_differentials(term)
    diff_vars = Symbol[]
    curr = term
    while SymbolicUtils.iscall(curr) && (SymbolicUtils.operation(curr) isa Differential)
        diff_op = SymbolicUtils.operation(curr)
        d_order = hasproperty(diff_op, :order) ? diff_op.order : 1
        for _ in 1:d_order
            push!(diff_vars, nameof(diff_op.x))
        end
        curr = SymbolicUtils.arguments(curr)[1]
    end
    return curr, diff_vars
end

function has_fixed_argument_depvar_derivative(term, dict_depvars)
    if !SymbolicUtils.iscall(term)
        return false
    end

    op = SymbolicUtils.operation(term)
    if op isa Differential
        inner_dv_term, _ = unwrap_differentials(term)
        if SymbolicUtils.iscall(inner_dv_term)
            dv_name = nameof(SymbolicUtils.operation(inner_dv_term))
            if haskey(dict_depvars, dv_name)
                return any(SymbolicUtils.arguments(inner_dv_term)) do arg
                    arg_ex = toexpr(arg)
                    arg_ex isa Number || Symbolics.value(arg) isa Number
                end
            end
        end
    end

    return any(
        arg -> has_fixed_argument_depvar_derivative(arg, dict_depvars),
        SymbolicUtils.arguments(term)
    )
end

function normalize_equation_residual(eq, dict_depvars)
    raw_residual = eq.lhs - eq.rhs

    # `expand_derivatives` is needed for product-rule PDE residuals, but it can
    # simplify Neumann BCs such as Dx(u(t, 1)) to zero because `x` is fixed in
    # the dependent-variable call. Preserve those direct boundary derivatives so
    # the parser can lower them to finite-difference stencil evaluations.
    if has_fixed_argument_depvar_derivative(unwrap(raw_residual), dict_depvars)
        return raw_residual
    end

    return expand_derivatives(raw_residual)
end

function coordinate_symbol(v, dict_indvars)
    v isa Number && return nothing
    s = try
        v isa Symbol ? v : Symbolics.tosymbol(v)
    catch
        return nothing
    end
    return (s isa Symbol && haskey(dict_indvars, s)) ? s : nothing
end

function coordinate_index_map(layout, dict_indvars)
    pairs = Pair{Symbol, Int}[]
    for (i, v) in enumerate(layout)
        s = coordinate_symbol(v, dict_indvars)
        s === nothing && continue
        push!(pairs, s => i)
    end
    return Dict(pairs)
end

function equation_coordinate_layout(eq, dict_indvars, dict_depvars)
    args = get_argument([eq], dict_indvars, dict_depvars)
    return isempty(args) ? Any[] : args[1]
end

function is_global_coordinate_map(dict_cord_indvars, dict_indvars)
    length(dict_cord_indvars) == length(dict_indvars) || return false
    return all(p -> get(dict_cord_indvars, first(p), 0) == last(p), dict_indvars)
end

function local_coordinate_index_map(eq, dict_indvars, dict_depvars, strategy, local_indvars, bcs)
    eq_layout = equation_coordinate_layout(eq, dict_indvars, dict_depvars)
    is_bc = any(bc -> isequal(eq, bc), bcs)

    # Grid/stochastic/quasirandom training data keeps the equation argument
    # layout, including fixed numeric boundary rows. Quadrature BC bounds are
    # compact and include only free symbolic variables.
    runtime_layout = if strategy isa QuadratureTraining && is_bc
        source_layout = local_indvars === nothing ? eq_layout : local_indvars
        filter(v -> coordinate_symbol(v, dict_indvars) !== nothing, source_layout)
    else
        eq_layout
    end

    if isempty(runtime_layout) && local_indvars !== nothing
        runtime_layout = local_indvars
    end

    return coordinate_index_map(runtime_layout, dict_indvars)
end

function build_pinn_rewriter(
        dict_depvars, dict_depvar_input, dict_indvars, cord_sym, phi_sym, θ_sym, derivative_sym,
        is_multioutput; dict_cord_indvars = dict_indvars,
        dict_params = Dict{Symbol, Int}(), param_defaults = Float64[]
    )
    is_reduced_cord = !is_global_coordinate_map(dict_cord_indvars, dict_indvars)

    rule = function (t)
        if !SymbolicUtils.iscall(t)
            ex = toexpr(t)
            if ex isa Symbol && haskey(dict_cord_indvars, ex)
                row_idx = dict_cord_indvars[ex]
                return SymbolicUtils.term(cord_slice, cord_sym, row_idx; type = Real)
            elseif ex isa Symbol && haskey(dict_params, ex)
                param_idx = dict_params[ex]
                default_val = param_idx <= length(param_defaults) ? param_defaults[param_idx] : NaN
                return SymbolicUtils.term(pde_param_value, θ_sym, param_idx, default_val; type = Real)
            end
            return t
        end

        op = SymbolicUtils.operation(t)

        if op isa Differential
            inner_dv_term, diff_vars = unwrap_differentials(t)
            if SymbolicUtils.iscall(inner_dv_term)
                dv_name = nameof(SymbolicUtils.operation(inner_dv_term))
                if haskey(dict_depvars, dv_name)
                    dv_idx = dict_depvars[dv_name]
                    dv_indvars = dict_depvar_input[dv_name]
                    directions = map(diff_vars) do v
                        dir = findfirst(==(v), dv_indvars)
                        dir === nothing && throw(ArgumentError("Cannot lower derivative with respect to $(v) for dependent variable $(dv_name) with inputs $(dv_indvars)."))
                        dir
                    end

                    cord_slice_term = if !is_reduced_cord
                        global_row_indices = [dict_indvars[v] for v in dv_indvars]
                        if length(global_row_indices) == length(dict_indvars) && global_row_indices == collect(1:length(dict_indvars))
                            cord_sym
                        elseif length(global_row_indices) == 1
                            SymbolicUtils.term(cord_slice, cord_sym, global_row_indices[1]; type = Real)
                        else
                            SymbolicUtils.term(cord_slice_rows, cord_sym, global_row_indices; type = Real)
                        end
                    else
                        dv_args = SymbolicUtils.arguments(inner_dv_term)
                        row_types = Int[]
                        row_indices = Int[]
                        row_consts = Float64[]
                        is_identity_cord = true

                        for (k, arg) in enumerate(dv_args)
                            arg_sym = Symbolics.tosymbol(arg)
                            if arg_sym isa Symbol && haskey(dict_cord_indvars, arg_sym)
                                idx = dict_cord_indvars[arg_sym]
                                push!(row_types, 1)
                                push!(row_indices, idx)
                                push!(row_consts, 0.0)
                                if idx != k || length(dict_cord_indvars) != length(dv_args)
                                    is_identity_cord = false
                                end
                            else
                                v = Symbolics.value(arg)
                                num_val = (v isa Number) ? Float64(v) : 0.0
                                push!(row_types, 0)
                                push!(row_indices, 0)
                                push!(row_consts, num_val)
                                is_identity_cord = false
                            end
                        end

                        if is_identity_cord
                            cord_sym
                        else
                            SymbolicUtils.term(construct_bc_cord, cord_sym, row_types, row_indices, row_consts; type = Real)
                        end
                    end

                    return SymbolicUtils.term(deriv_fd, derivative_sym, phi_sym, cord_slice_term, θ_sym, directions, dv_idx, is_multioutput; type = Real)
                end
            end
        end

        if op isa Symbolics.Integral
            domain = op.domain
            int_vars = domain.variables
            int_var_tuple = if SymbolicUtils.iscall(int_vars) && (nameof(SymbolicUtils.operation(int_vars)) === :tuple || SymbolicUtils.operation(int_vars) === tuple)
                SymbolicUtils.arguments(int_vars)
            elseif int_vars isa Tuple || int_vars isa AbstractVector
                collect(int_vars)
            else
                [int_vars]
            end
            int_var_indices = [dict_indvars[Symbolics.tosymbol(v)] for v in int_var_tuple]

            lb, ub = get_limits(domain.domain)

            is_inf_neg(x) = begin
                s = Symbolics.tosymbol(x)
                (s isa Number && s == -Inf) || s === Symbol("-Inf")
            end
            is_inf_pos(x) = begin
                s = Symbolics.tosymbol(x)
                (s isa Number && s == Inf) || s === :Inf || s === Symbol("Inf")
            end
            to_float(x) = begin
                v = Symbolics.value(x)
                (v isa Number) ? Float64(v) : 0.0
            end

            is_inf_vec = Int[
                if is_inf_neg(l) && is_inf_pos(u)
                    2
                elseif is_inf_pos(u)
                    1
                else
                    0
                end
                    for (l, u) in zip(lb, ub)
            ]

            lb_vals = Float64[to_float(l) for l in lb]
            ub_vals = Float64[to_float(u) for u in ub]
            lb_col_indices = Int[haskey(dict_cord_indvars, toexpr(l)) ? dict_cord_indvars[toexpr(l)] : 0 for l in lb]
            ub_col_indices = Int[haskey(dict_cord_indvars, toexpr(u)) ? dict_cord_indvars[toexpr(u)] : 0 for u in ub]

            integrand_term = SymbolicUtils.arguments(t)[1]
            rewritten_integrand = SymbolicUtils.Rewriters.Prewalk(rule)(integrand_term)
            integrand_expr_raw = Symbolics.build_function(rewritten_integrand, cord_sym, phi_sym, θ_sym, derivative_sym; expression = Val{true}, cse = true)
            integrand_fn = @RuntimeGeneratedFunction(_dot_(integrand_expr_raw))

            return SymbolicUtils.term(eval_numeric_integral, integrand_fn, cord_sym, phi_sym, θ_sym, derivative_sym, int_var_indices, lb_vals, ub_vals, lb_col_indices, ub_col_indices, is_inf_vec; type = Real)
        end

        op_ex = toexpr(op)
        if op_ex isa Symbol && haskey(dict_depvars, op_ex)
            dv_idx = dict_depvars[op_ex]
            dv_indvars = dict_depvar_input[op_ex]

            cord_slice_term = if !is_reduced_cord
                global_row_indices = [dict_indvars[v] for v in dv_indvars]
                if length(global_row_indices) == length(dict_indvars) && global_row_indices == collect(1:length(dict_indvars))
                    cord_sym
                elseif length(global_row_indices) == 1
                    SymbolicUtils.term(cord_slice, cord_sym, global_row_indices[1]; type = Real)
                else
                    SymbolicUtils.term(cord_slice_rows, cord_sym, global_row_indices; type = Real)
                end
            else
                dv_args = SymbolicUtils.arguments(t)
                row_types = Int[]
                row_indices = Int[]
                row_consts = Float64[]
                is_identity_cord = true

                for (k, arg) in enumerate(dv_args)
                    arg_sym = Symbolics.tosymbol(arg)
                    if arg_sym isa Symbol && haskey(dict_cord_indvars, arg_sym)
                        idx = dict_cord_indvars[arg_sym]
                        push!(row_types, 1)
                        push!(row_indices, idx)
                        push!(row_consts, 0.0)
                        if idx != k || length(dict_cord_indvars) != length(dv_args)
                            is_identity_cord = false
                        end
                    else
                        v = Symbolics.value(arg)
                        num_val = (v isa Number) ? Float64(v) : 0.0
                        push!(row_types, 0)
                        push!(row_indices, 0)
                        push!(row_consts, num_val)
                        is_identity_cord = false
                    end
                end

                if is_identity_cord
                    cord_sym
                else
                    SymbolicUtils.term(construct_bc_cord, cord_sym, row_types, row_indices, row_consts; type = Real)
                end
            end

            return SymbolicUtils.term(phi_eval, phi_sym, cord_slice_term, θ_sym, dv_idx, is_multioutput; type = Real)
        end

        return t
    end

    return SymbolicUtils.Rewriters.Prewalk(rule)
end

# ----------------------------------------------------------------------
# 3. Vectorization & Code Generation
# ----------------------------------------------------------------------

"""
    build_batched_symbolic_loss_function(pinnrep, eq; bc_indvars = nothing, ...)

High-level entry point that translates a PDE or boundary condition equation into a 
compiled, batched matrix loss kernel function using the SymbolicUtils parser.
"""
function build_batched_symbolic_loss_function(pinnrep::PINNRepresentation, eq; bc_indvars = nothing, kwargs...)
    (;
        bcs, depvars, dict_depvars, dict_depvar_input, dict_indvars, multioutput, phi,
        derivative, strategy, eq_params, default_p,
    ) = pinnrep

    cord_sym = unwrap(Symbolics.variable(:__cord__))
    phi_sym = unwrap(Symbolics.variable(:__phi__))
    θ_sym = unwrap(Symbolics.variable(:__θ__))
    derivative_sym = unwrap(Symbolics.variable(:__derivative__))

    is_multioutput = (multioutput isa Bool) ? multioutput : (length(depvars) > 1)

    clean_eq_params = eq_params isa SciMLBase.NullParameters ? [] : collect(eq_params)
    dict_params = Dict{Symbol, Int}(
        Symbolics.tosymbol(p) => i for (i, p) in enumerate(clean_eq_params)
    )
    param_defaults = default_p === nothing ? Float64[] : Float64.(default_p)

    dict_cord_indvars = local_coordinate_index_map(
        eq, dict_indvars, dict_depvars, strategy, bc_indvars, bcs
    )

    # 1. Expand the full residual first, then transform via Prewalk.
    rewriter = build_pinn_rewriter(
        dict_depvars, dict_depvar_input, dict_indvars, cord_sym, phi_sym, θ_sym, derivative_sym,
        is_multioutput; dict_cord_indvars = dict_cord_indvars,
        dict_params = dict_params, param_defaults = param_defaults
    )

    residual = normalize_equation_residual(eq, dict_depvars)
    loss_kernel_sym = rewriter(unwrap(residual))

    # 2. Build batched function with Common Subexpression Elimination (CSE)
    expr_raw = Symbolics.build_function(
        loss_kernel_sym, cord_sym, phi_sym, θ_sym, derivative_sym;
        expression = Val{true},
        cse = true
    )

    # 3. Apply broadcasting to arithmetic operations via _dot_
    expr_dotted = _dot_(expr_raw)

    # 4. Return compiled runtime-generated function
    compiled_kernel = @RuntimeGeneratedFunction(expr_dotted)
    return (cord, θ) -> compiled_kernel(cord, phi, θ, derivative)
end
