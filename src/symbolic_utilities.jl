"""
Variable extraction, domain decomposition, and symbolic inspection utilities for NeuralPDE.
"""

"""
Create dictionary: variable => unique number for variable

## Example
Dict{Symbol,Int64} with 3 entries:
  :x => 1
  :y => 2
  :t => 3
"""
get_dict_vars(vars) = Dict([Symbol(v) .=> i for (i, v) in enumerate(vars)])

function get_ε(dim::Int, der_num::Int, ::Type{eltypeθ}, order) where {eltypeθ}
    epsilon = ^(eps(eltypeθ), one(eltypeθ) / (2 + order))
    ε = zeros(eltypeθ, dim)
    ε[der_num] = epsilon
    return ε
end

@inline function get_fd_step(::Type{T}, order::Int) where {T}
    return eps(T)^(one(T) / (T(2) + T(order)))
end

@inline get_fd_step(x::AbstractArray{T}, order::Int) where {T} = get_fd_step(T, order)

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

function phi_eval(phi, cord_batch::AbstractMatrix, θ, dv_idx::Int, is_multioutput::Bool)
    θ_dv = get_pinn_theta(θ, dv_idx, is_multioutput)
    phi_dv = (is_multioutput && (phi isa Tuple || phi isa AbstractVector)) ? phi[dv_idx] : phi
    return phi_dv(cord_batch, θ_dv)
end

function deriv_fd(
        derivative, phi, cord_batch::AbstractMatrix{T}, θ, directions::Vector{Int},
        dv_idx::Int, is_multioutput::Bool
    ) where {T}
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

function eval_numeric_integral(
        integrand_fn, cord_batch::AbstractMatrix{T}, phi, θ, derivative,
        int_var_indices::Vector{Int}, lb_vals::Vector{Float64},
        ub_vals::Vector{Float64}, lb_col_indices::Vector{Int},
        ub_col_indices::Vector{Int}, is_inf_vec::Vector{Int}
    ) where {T}
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
                prob = Integrals.IntegralProblem(
                    integrand, (zero(T), one(T) - T(1.0e-4)), θ
                )
                sol = Integrals.solve(
                    prob, Integrals.QuadGKJL(); reltol = 1.0e-4, abstol = 1.0e-4
                )
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
                prob = Integrals.IntegralProblem(
                    integrand,
                    (-one(T) + T(1.0e-4), one(T) - T(1.0e-4)), θ
                )
                sol = Integrals.solve(
                    prob, Integrals.QuadGKJL(); reltol = 1.0e-4, abstol = 1.0e-4
                )
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
                sol = Integrals.solve(
                    prob, Integrals.QuadGKJL(); reltol = 1.0e-4, abstol = 1.0e-4
                )
                return sol.u
            end
        else
            lb_pt = @ignore_derivatives T[
                (lb_col_indices[k] > 0) ? col[lb_col_indices[k]] : lb_vals[k]
                    for k in 1:K
            ]
            ub_pt = @ignore_derivatives T[
                (ub_col_indices[k] > 0) ? col[ub_col_indices[k]] : ub_vals[k]
                    for k in 1:K
            ]

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
            sol = Integrals.solve(
                prob, Integrals.HCubatureJL(); reltol = 1.0e-3, abstol = 1.0e-3
            )
            return sol.u
        end
    end

    return reshape(T.(sol_array), 1, N)
end

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

function local_coordinate_index_map(eq, dict_indvars, dict_depvars, strategy, local_indvars, bcs)
    eq_layout = equation_coordinate_layout(eq, dict_indvars, dict_depvars)
    is_bc = any(bc -> isequal(eq, bc), bcs)

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

function get_limits(domain)
    if domain isa AbstractInterval
        return [leftendpoint(domain)], [rightendpoint(domain)]
    elseif domain isa ProductDomain
        return collect(map(leftendpoint, DomainSets.components(domain))),
            collect(map(rightendpoint, DomainSets.components(domain)))
    end
end

function get_vars(indvars_, depvars_)
    indvars = SymbolicIndexingInterface.getname.(indvars_)
    depvars = Symbol[]
    dict_depvar_input = Dict{Symbol, Vector{Symbol}}()
    for d in depvars_
        if SymbolicUtils.iscall(unwrap(d))
            dname = SymbolicIndexingInterface.getname(d)
            push!(depvars, dname)
            push!(
                dict_depvar_input,
                dname => [
                    nameof(unwrap(argument))
                        for argument in arguments(unwrap(d))
                ]
            )
        else
            dname = SymbolicIndexingInterface.getname(d)
            push!(depvars, dname)
            push!(dict_depvar_input, dname => indvars) # default to all inputs if not given
        end
    end

    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input
end

function get_integration_variables(eqs, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars,
        dict_depvar_input = get_vars(
        _indvars,
        _depvars
    )
    return get_integration_variables(eqs, dict_indvars, dict_depvars)
end

function get_integration_variables(eqs, dict_indvars, dict_depvars)
    exprs = toexpr.(eqs)
    return vars = map(exprs) do expr
        _vars = Symbol.(
            filter(
                indvar -> length(find_thing_in_expr(expr, indvar)) > 0,
                sort(collect(keys(dict_indvars)))
            )
        )
    end
end

"""
    get_variables(eqs, _indvars, _depvars)

Returns all variables that are used in each equations or boundary condition.
"""
function get_variables end

function get_variables(eqs, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars,
        dict_depvar_input = get_vars(
        _indvars,
        _depvars
    )
    return get_variables(eqs, dict_indvars, dict_depvars)
end

function get_variables(eqs, dict_indvars, dict_depvars)
    bc_args = get_argument(eqs, dict_indvars, dict_depvars)
    return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
end

function get_number(eqs, dict_indvars, dict_depvars)
    bc_args = get_argument(eqs, dict_indvars, dict_depvars)
    return map(barg -> filter(x -> x isa Number, barg), bc_args)
end

function find_thing_in_expr(ex::Expr, thing; ans = [])
    if thing in ex.args
        push!(ans, ex)
    end
    for e in ex.args
        if e isa Expr
            if thing in e.args
                push!(ans, e)
            end
            find_thing_in_expr(e, thing; ans = ans)
        end
    end
    return collect(Set(ans))
end

"""
    get_argument(eqs, _indvars::Array, _depvars::Array)

Returns all arguments that are used in each equations or boundary condition.
"""
function get_argument end

function get_argument(eqs, _indvars::Array, _depvars::Array)
    _, _, dict_indvars, dict_depvars, _ = get_vars(_indvars, _depvars)
    return get_argument(eqs, dict_indvars, dict_depvars)
end

function get_argument(eqs, dict_indvars, dict_depvars)
    exprs = toexpr.(eqs)
    vars = map(exprs) do expr
        _vars = map(depvar -> find_thing_in_expr(expr, depvar), collect(keys(dict_depvars)))
        f_vars = filter(x -> !isempty(x), _vars)
        map(x -> first(x), f_vars)
    end
    args_ = map(vars) do _vars
        ind_args_ = map(var -> var.args[2:end], _vars)
        syms = Set{Symbol}()
        filter(vcat(ind_args_...)) do ind_arg
            if ind_arg isa Symbol
                if ind_arg ∈ syms
                    false
                else
                    push!(syms, ind_arg)
                    true
                end
            else
                true
            end
        end
    end
    return args_
end
