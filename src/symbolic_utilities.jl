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

const NON_DOTTABLE_SYMBOLS = Set(
    [
        :cord_slice, :cord_slice_rows, :construct_bc_cord, :phi_eval, :deriv_fd, :eval_numeric_integral,
        :get_pinn_theta, :pde_param_value, :vector_to_parameters, :Pair, :Dict, :tuple, :(:),
        :size, :length, :zeros, :ones, :fill, :reshape, :copy, :Array, :Vector, :Matrix,
    ]
)

function get_op_symbol(fn)
    if fn isa QuoteNode
        return get_op_symbol(fn.value)
    elseif fn isa Function
        return nameof(fn)
    elseif fn isa Symbol
        return fn
    elseif fn isa GlobalRef
        return fn.name
    elseif fn isa Expr
        if fn.head === :$ && length(fn.args) == 1
            return get_op_symbol(fn.args[1])
        end
        if (fn.head === :. || fn.head === :('.')) && length(fn.args) >= 2
            second = fn.args[2]
            return second isa QuoteNode ? second.value : (second isa Symbol ? second : nothing)
        end
    end
    return nothing
end

function dottable_(x)
    sym = get_op_symbol(x)
    if sym !== nothing
        return !(sym in NON_DOTTABLE_SYMBOLS) && Base.Broadcast.dottable(sym)
    end
    return false
end

function integer_power_exponent(x, const_env = Dict{Any, Any}())
    x isa QuoteNode && return integer_power_exponent(x.value, const_env)
    if haskey(const_env, x)
        return integer_power_exponent(const_env[x], const_env)
    end
    x isa Integer && return x
    if x isa AbstractFloat && isfinite(x) && isinteger(x)
        return Int(x)
    end
    return nothing
end

is_power_operator(x) = get_op_symbol(x) in (:^, :pow)

function dotted_literal_power(base, exponent, const_env = Dict{Any, Any}())
    return Expr(
        :.,
        Expr(:., :Base, QuoteNode(:literal_pow)),
        Expr(:tuple, :^, _dot_(base, const_env), Expr(:call, :Val, exponent))
    )
end

function dotted_power_rewrite(x::Expr, const_env = Dict{Any, Any}())
    if x.head === :call && length(x.args) == 3 && is_power_operator(x.args[1])
        exponent = integer_power_exponent(x.args[3], const_env)
        exponent !== nothing && return dotted_literal_power(x.args[2], exponent, const_env)
    elseif x.head === :. && length(x.args) == 2 && is_power_operator(x.args[1]) &&
            Meta.isexpr(x.args[2], :tuple) && length(x.args[2].args) == 2
        exponent = integer_power_exponent(x.args[2].args[2], const_env)
        exponent !== nothing && return dotted_literal_power(x.args[2].args[1], exponent, const_env)
    end
    return nothing
end

"""
    _dot_(ex)

Recursively transforms function calls in Julia `Expr` ASTs into dot-broadcasted calls (e.g. `.+`, `.*`, `sin.`)
while preserving matrix runtime kernels (such as `cord_slice`, `deriv_fd`, `eval_numeric_integral`) and `:let` blocks.
"""
_dot_(x) = x
function _dot_(x, const_env::Dict{Any, Any})
    return x
end

function _dot_(x::Expr)
    return _dot_(x, Dict{Any, Any}())
end

function _dot_(x::Expr, const_env::Dict{Any, Any})
    power_rewrite = dotted_power_rewrite(x, const_env)
    if power_rewrite !== nothing
        return power_rewrite
    elseif x.head === :call && dottable_(x.args[1])
        dotargs = Base.mapany(arg -> _dot_(arg, const_env), x.args[2:end])
        return Expr(:., x.args[1], Expr(:tuple, dotargs...))
    elseif x.head === :call
        dotargs = Base.mapany(arg -> _dot_(arg, const_env), x.args[2:end])
        return Expr(:call, x.args[1], dotargs...)
    elseif x.head === :comparison
        dotargs = Base.mapany(arg -> _dot_(arg, const_env), x.args)
        return Expr(
            :comparison,
            (
                iseven(i) && dottable_(arg) && arg isa Symbol ?
                    Symbol('.', arg) : arg for (i, arg) in pairs(dotargs)
            )...
        )
    elseif x.head === :$
        return x.args[1]
    elseif x.head === :(=) && Meta.isexpr(x.args[1], :call) # function definition
        return Expr(x.head, x.args[1], _dot_(x.args[2], const_env))
    elseif x.head === :(=) # assignment (e.g. let binding)
        rhs = _dot_(x.args[2], const_env)
        if x.args[1] isa Symbol
            exponent = integer_power_exponent(rhs, const_env)
            if exponent === nothing
                delete!(const_env, x.args[1])
            else
                const_env[x.args[1]] = exponent
            end
        end
        return Expr(:(=), x.args[1], rhs)
    elseif x.head === :let # let bindings block
        local_env = copy(const_env)
        return Expr(:let, _dot_(x.args[1], local_env), _dot_(x.args[2], local_env))
    elseif x.head === :for
        return Expr(:for, x.args[1], _dot_(x.args[2], copy(const_env)))
    elseif x.head === :block
        local_env = copy(const_env)
        return Expr(:block, Base.mapany(arg -> _dot_(arg, local_env), x.args)...)
    else
        dotargs = Base.mapany(arg -> _dot_(arg, const_env), x.args)
        return Expr(x.head, dotargs...)
    end
end
