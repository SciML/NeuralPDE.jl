using Base.Broadcast

"""
Override `Broadcast.__dot__` with `Broadcast.dottable(x::Function) = true`

## Example

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

## Examples

1. First compute the derivative of function 'u(x,y)' with respect to x.

Take expressions in the form: `derivative(u(x,y), x)` to `derivative(phi, u, [x, y], εs, order, θ)`,
where
- phi - trial solution.
- u - function.
- x,y - coordinates of point.
- εs - epsilon mask.
- order - order of derivative.
- θ - weights in neural network.
"""
function _transform_expression(pinnrep::PINNRepresentation, ex; is_integral = false,
                               dict_transformation_vars = nothing,
                               transformation_vars = nothing)
    @unpack indvars, depvars, dict_indvars, dict_depvars,
    dict_depvar_input, multioutput, strategy, phi,
    derivative, integral, flat_init_params, init_params = pinnrep
    eltypeθ = eltype(flat_init_params)

    _args = ex.args
    for (i, e) in enumerate(_args)
        if !(e isa Expr)
            if e in keys(dict_depvars)
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                var_ = is_integral ? :(u) : :($(Expr(:$, :u)))
                ex.args = if !multioutput
                    [var_, Symbol(:cord, num_depvar), :($θ), :phi]
                else
                    [
                        var_,
                        Symbol(:cord, num_depvar),
                        Symbol(:($θ), num_depvar),
                        Symbol(:phi, num_depvar),
                    ]
                end
                break
            elseif e isa ModelingToolkit.Differential
                derivative_variables = Symbol[]
                order = 0
                while (_args[1] isa ModelingToolkit.Differential)
                    order += 1
                    push!(derivative_variables, toexpr(_args[1].x))
                    _args = _args[2].args
                end
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                dict_interior_indvars = Dict([indvar .=> j
                                              for (j, indvar) in enumerate(dict_depvar_input[depvar])])
                dim_l = length(dict_interior_indvars)

                var_ = is_integral ? :(derivative) : :($(Expr(:$, :derivative)))
                εs = [get_ε(dim_l, d, eltypeθ, order) for d in 1:dim_l]
                undv = [dict_interior_indvars[d_p] for d_p in derivative_variables]
                εs_dnv = [εs[d] for d in undv]

                ex.args = if !multioutput
                    [var_, :phi, :u, Symbol(:cord, num_depvar), εs_dnv, order, :($θ)]
                else
                    [
                        var_,
                        Symbol(:phi, num_depvar),
                        :u,
                        Symbol(:cord, num_depvar),
                        εs_dnv,
                        order,
                        Symbol(:($θ), num_depvar),
                    ]
                end
                break
            elseif e isa Symbolics.Integral
                if _args[1].domain.variables isa Tuple
                    integrating_variable_ = collect(_args[1].domain.variables)
                    integrating_variable = toexpr.(integrating_variable_)
                    integrating_var_id = [dict_indvars[i] for i in integrating_variable]
                else
                    integrating_variable = toexpr(_args[1].domain.variables)
                    integrating_var_id = [dict_indvars[integrating_variable]]
                end

                integrating_depvars = []
                integrand_expr = _args[2]
                for d in depvars
                    d_ex = find_thing_in_expr(integrand_expr, d)
                    if !isempty(d_ex)
                        push!(integrating_depvars, d_ex[1].args[1])
                    end
                end

                lb, ub = get_limits(_args[1].domain.domain)
                lb, ub, _args[2], dict_transformation_vars, transformation_vars = transform_inf_integral(lb,
                                                                                                         ub,
                                                                                                         _args[2],
                                                                                                         integrating_depvars,
                                                                                                         dict_depvar_input,
                                                                                                         dict_depvars,
                                                                                                         integrating_variable,
                                                                                                         eltypeθ)

                num_depvar = map(int_depvar -> dict_depvars[int_depvar],
                                 integrating_depvars)
                integrand_ = transform_expression(pinnrep, _args[2];
                                                  is_integral = false,
                                                  dict_transformation_vars = dict_transformation_vars,
                                                  transformation_vars = transformation_vars)
                integrand__ = _dot_(integrand_)

                integrand = build_symbolic_loss_function(pinnrep, nothing;
                                                         integrand = integrand__,
                                                         integrating_depvars = integrating_depvars,
                                                         eq_params = SciMLBase.NullParameters(),
                                                         dict_transformation_vars = dict_transformation_vars,
                                                         transformation_vars = transformation_vars,
                                                         param_estim = false,
                                                         default_p = nothing)
                # integrand = repr(integrand)
                lb = toexpr.(lb)
                ub = toexpr.(ub)
                ub_ = []
                lb_ = []
                for l in lb
                    if l isa Number
                        push!(lb_, l)
                    else
                        l_expr = NeuralPDE.build_symbolic_loss_function(pinnrep, nothing;
                                                                        integrand = _dot_(l),
                                                                        integrating_depvars = integrating_depvars,
                                                                        param_estim = false,
                                                                        default_p = nothing)
                        l_f = @RuntimeGeneratedFunction(l_expr)
                        push!(lb_, l_f)
                    end
                end
                for u_ in ub
                    if u_ isa Number
                        push!(ub_, u_)
                    else
                        u_expr = NeuralPDE.build_symbolic_loss_function(pinnrep, nothing;
                                                                        integrand = _dot_(u_),
                                                                        integrating_depvars = integrating_depvars,
                                                                        param_estim = false,
                                                                        default_p = nothing)
                        u_f = @RuntimeGeneratedFunction(u_expr)
                        push!(ub_, u_f)
                    end
                end

                integrand_func = @RuntimeGeneratedFunction(integrand)
                ex.args = [
                    :($(Expr(:$, :integral))),
                    :u,
                    Symbol(:cord, num_depvar[1]),
                    :phi,
                    integrating_var_id,
                    integrand_func,
                    lb_,
                    ub_,
                    :($θ),
                ]
                break
            end
        else
            ex.args[i] = _transform_expression(pinnrep, ex.args[i];
                                               is_integral = is_integral,
                                               dict_transformation_vars = dict_transformation_vars,
                                               transformation_vars = transformation_vars)
        end
    end
    return ex
end

"""
Parse ModelingToolkit equation form to the inner representation.

## Examples:

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

function get_vars(indvars_, depvars_)
    indvars = ModelingToolkit.getname.(indvars_)
    depvars = Symbol[]
    dict_depvar_input = Dict{Symbol, Vector{Symbol}}()
    for d in depvars_
        if unwrap(d) isa SymbolicUtils.BasicSymbolic
            dname = ModelingToolkit.getname(d)
            push!(depvars, dname)
            push!(dict_depvar_input,
                  dname => [nameof(unwrap(argument))
                            for argument in arguments(unwrap(d))])
        else
            dname = ModelingToolkit.getname(d)
            push!(depvars, dname)
            push!(dict_depvar_input, dname => indvars) # default to all inputs if not given
        end
    end

    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input
end

function get_integration_variables(eqs, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
    get_integration_variables(eqs, dict_indvars, dict_depvars)
end

function get_integration_variables(eqs, dict_indvars, dict_depvars)
    exprs = toexpr.(eqs)
    vars = map(exprs) do expr
        _vars = Symbol.(filter(indvar -> length(find_thing_in_expr(expr, indvar)) > 0,
                               sort(collect(keys(dict_indvars)))))
    end
end

"""
    get_variables(eqs,_indvars,_depvars)

Returns all variables that are used in each equations or boundary condition.
"""
function get_variables end

function get_variables(eqs, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
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
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
    get_argument(eqs, dict_indvars, dict_depvars)
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
