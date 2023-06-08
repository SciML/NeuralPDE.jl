using Base.Broadcast

function get_limits(domain)
    if domain isa AbstractInterval
        return [leftendpoint(domain)], [rightendpoint(domain)]
    elseif domain isa ProductDomain
        return collect(map(leftendpoint, DomainSets.components(domain))),
               collect(map(rightendpoint, DomainSets.components(domain)))
    end
end

θ = gensym("θ")

"""
Override `Broadcast.__dot__` with `Broadcast.dottable(x::Function) = true`

# Example

```julia
julia> e = :(1 + $sin(x))
:(1 + (sin)(x))

julia> Broadcast.__dot__(e)
:((+).(1, (sin)(x)))

julia> _dot_(e)
:((+).(1, (sin).(x)))
```
"""
dottable_(x) = Broadcast.dottable(x)
dottable_(x::Function) = true

_dot_(x) = x
function _dot_(x::Expr)
    dotargs = Base.mapany(_dot_, x.args)
    if x.head === :call && dottable_(x.args[1])
        Expr(:., dotargs[1], Expr(:tuple, dotargs[2:end]...))
    elseif x.head === :comparison
        Expr(:comparison,
             (iseven(i) && dottable_(arg) && arg isa Symbol && isoperator(arg) ?
              Symbol('.', arg) : arg for (i, arg) in pairs(dotargs))...)
    elseif x.head === :$
        x.args[1]
    elseif x.head === :let # don't add dots to `let x=...` assignments
        Expr(:let, undot(dotargs[1]), dotargs[2])
    elseif x.head === :for # don't add dots to for x=... assignments
        Expr(:for, undot(dotargs[1]), dotargs[2])
    elseif (x.head === :(=) || x.head === :function || x.head === :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], dotargs[2])
    elseif x.head === :(<:) || x.head === :(>:)
        tmp = x.head === :(<:) ? :.<: : :.>:
        Expr(:call, tmp, dotargs...)
    else
        head = String(x.head)::String
        if last(head) == '=' && first(head) != '.' || head == "&&" || head == "||"
            Expr(Symbol('.', head), dotargs...)
        else
            Expr(x.head, dotargs...)
        end
    end
end
"""
Create dictionary: variable => unique number for variable

# Example 1

Dict{Symbol,Int64} with 3 entries:
  :y => 2
  :t => 3
  :x => 1

# Example 2

 Dict{Symbol,Int64} with 2 entries:
  :u1 => 1
  :u2 => 2
"""
get_dict_vars(vars) = Dict([Symbol(v) .=> i for (i, v) in enumerate(vars)])

# Wrapper for _transform_expression
function transform_expression(pinnrep::PINNRepresentation, ex; is_integral = false,
                              dict_transformation_vars = nothing,
                              transformation_vars = nothing)
    if ex isa Expr
        ex = _transform_expression(pinnrep, ex; is_integral = is_integral,
                                   dict_transformation_vars = dict_transformation_vars,
                                   transformation_vars = transformation_vars)
    end
    return ex
end

function get_ε(dim::Int, der_num::Int, ::Type{eltypeθ}, order) where {eltypeθ}
    epsilon = ^(eps(eltypeθ), one(eltypeθ) / (2 + order))
    ε = zeros(eltypeθ, dim)
    ε[der_num] = epsilon
    ε
end

function get_limits(domain)
    if domain isa AbstractInterval
        return [leftendpoint(domain)], [rightendpoint(domain)]
    elseif domain isa ProductDomain
        return collect(map(leftendpoint, DomainSets.components(domain))),
               collect(map(rightendpoint, DomainSets.components(domain)))
    end
end

θ = gensym("θ")

"""
Transform the derivative expression to inner representation

# Examples

1. First compute the derivative of function 'u(x,y)' with respect to x.

Take expressions in the form: `derivative(u(x,y), x)` to `derivative(phi, u, [x, y], εs, order, θ)`,
where
 phi - trial solution
 u - function
 x,y - coordinates of point
 εs - epsilon mask
 order - order of derivative
 θ - weight in neural network
"""

"""
Parse ModelingToolkit equation form to the inner representation.

Example:

1)  1-D ODE: Dt(u(t)) ~ t +1

    Take expressions in the form: 'Equation(derivative(u(t), t), t + 1)' to 'derivative(phi, u_d, [t], [[ε]], 1, θ) - (t + 1)'

2)  2-D PDE: Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    Take expressions in the form:
     Equation(derivative(derivative(u(x, y), x), x) + derivative(derivative(u(x, y), y), y), -(sin(πx)) * sin(πy))
    to
     (derivative(phi,u, [x, y], [[ε,0],[ε,0]], 2, θ) + derivative(phi, u, [x, y], [[0,ε],[0,ε]], 2, θ)) - -(sin(πx)) * sin(πy)

3)  System of PDEs: [Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
                    Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

    Take expressions in the form:
    2-element Array{Equation,1}:
        Equation(derivative(u1(x, y), x) + 4 * derivative(u2(x, y), y), ModelingToolkit.Constant(0))
        Equation(derivative(u2(x, y), x) + 9 * derivative(u1(x, y), y), ModelingToolkit.Constant(0))
    to
      [(derivative(phi1, u1, [x, y], [[ε,0]], 1, θ1) + 4 * derivative(phi2, u, [x, y], [[0,ε]], 1, θ2)) - 0,
       (derivative(phi2, u2, [x, y], [[ε,0]], 1, θ2) + 9 * derivative(phi1, u, [x, y], [[0,ε]], 1, θ1)) - 0]
"""
function parse_equation(pinnrep::PINNRepresentation, eq)
    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)
    left_expr = transform_expression(pinnrep, toexpr(eq_lhs))
    right_expr = transform_expression(pinnrep, toexpr(eq_rhs))
    left_expr = _dot_(left_expr)
    right_expr = _dot_(right_expr)
    loss_func = :($left_expr .- $right_expr)
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
    indvars_ex
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
    Dict(filter(p -> p !== nothing, pair_))
end

function get_integration_variables(eqs, v::VariableMap)
    ivs = all_ivs(v)
    return map(eq -> get_indvars(eq, ivs), eqs)
end

"""
```julia
get_argument(eqs,_indvars::Array,_depvars::Array)
```

Returns all arguments that are used in each equations or boundary condition.
"""
function get_argument end

function get_argument(eqs, v::VariableMap)
    vars = map(eqs) do eq
        _vars = map(depvar -> get_depvars(eq, [depvar]), v.depvar_ops)
        f_vars = filter(x -> !isempty(x), _vars)
        map(first, f_vars)
    end
    @show vars
    args_ = map(vars) do _vars
        seen = []
        filter(reduce(vcat, arguments.(_vars), init = [])) do x
            if x isa Number
                true
            else
                if any(isequal(x), seen)
                    false
                else
                    push!(seen, x)
                    true
                end
            end
        end
    end
    return args_ # TODO for all arguments
end

"""
``julia
get_variables(eqs,_indvars,_depvars)
```

Returns all variables that are used in each equations or boundary condition.
"""
function get_variables(eqs, v::VariableMap)
    args = get_argument(eqs, v)
    return map(arg -> filter(x -> !(x isa Number), arg), args)
end

function get_number(eqs, v::VariableMap)
    args = get_argument(eqs, v)
    return map(arg -> filter(x -> x isa Number, arg), args)
end
