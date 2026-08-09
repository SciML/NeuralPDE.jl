using Base.Broadcast

# build_expr was removed from Symbolics.jl v7; define locally
function build_expr(head::Symbol, args)
    ex = Expr(head)
    append!(ex.args, args)
    return ex
end

"""
Create dictionary: variable => unique number for variable

## Example 1

Dict{Symbol,Int64} with 3 entries:
  :y => 2
  :t => 3
  :x => 1

## Example 2

 Dict{Symbol,Int64} with 2 entries:
  :u1 => 1
  :u2 => 2
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

function get_indvars_ex(bc_indvars) # , dict_this_eq_indvars)
    i_ = 1
    indvars_ex = map(bc_indvars) do u
        if u isa Symbol
            # i = dict_this_eq_indvars[u]
            # ex = :($:cord[[$i],:])
            ex = :($:cord[[$i_], :])
            i_ += 1
            ex
        else
            :(fill($u, size($:cord[[1], :])))
        end
    end
    return indvars_ex
end

"""
Finds which dependent variables are being used in an equation.
"""
function pair(eq, depvars, dict_depvars, dict_depvar_input)
    expr = toexpr(eq)
    pair_ = map(depvars) do depvar
        if !isempty(find_thing_in_expr(expr, depvar))
            dict_depvars[depvar] => dict_depvar_input[depvar]
        end
    end
    return Dict(filter(p -> p !== nothing, pair_))
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
    get_variables(eqs,_indvars,_depvars)

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
    get_argument(eqs,_indvars::Array,_depvars::Array)

Returns all arguments that are used in each equations or boundary condition.
"""
function get_argument end

# Get arguments from boundary condition functions
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
    return args_ # TODO for all arguments
end
