using Base.Broadcast

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
        Expr(:comparison, (iseven(i) && dottable_(arg) && arg isa Symbol && isoperator(arg) ?
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

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
"""
struct LogOptions 
    log_frequency::Int64
    # TODO: add in an option for saving plots in the log. this is currently not done because the type of plot is dependent on the PDESystem
    #       possible solution: pass in a plot function?  
    #       this is somewhat important because we want to support plotting adaptive weights that depend on pde independent variables 
    #       and not just one weight for each loss function, i.e. pde_loss_weights(i, t, x) and since this would be function-internal, 
    #       we'd want the plot & log to happen internally as well
    #       plots of the learned function can happen in the outer callback, but we might want to offer that here too

    SciMLBase.@add_kwonly function LogOptions(;log_frequency=50)
        new(convert(Int64, log_frequency))
    end
end

"""This function is defined here as stubs to be overriden by the subpackage NeuralPDELogging if imported"""
function logvector(logger, v::AbstractVector{R}, name::AbstractString, step::Integer) where R <: Real
    nothing
end

"""This function is defined here as stubs to be overriden by the subpackage NeuralPDELogging if imported"""
function logscalar(logger, s::R, name::AbstractString, step::Integer) where R <: Real
    nothing
end


abstract type AbstractPINN end

"""
Algorithm for solving Physics-Informed Neural Networks problems.

Arguments:
* `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `strategy`: determines which training strategy will be used,
* `init_params`: the initial parameter of the neural network,
* `phi`: a trial solution,
* `derivative`: method that calculates the derivative.
"""
struct PhysicsInformedNN{T,P,PH,DER,PE,AL,ADA,LOG,K} <: AbstractPINN
  strategy::T
  init_params::P
  phi::PH
  derivative::DER
  param_estim::PE
  additional_loss::AL
  adaptive_loss::ADA
  logger::LOG
  log_options::LogOptions
  iteration::Vector{Int64}
  self_increment::Bool
  multioutput::Bool
  kwargs::K

  @add_kwonly function PhysicsInformedNN(chain,
                                         strategy;
                                         init_params = nothing,
                                         phi = nothing,
                                         derivative = nothing,
                                         param_estim=false,
                                         additional_loss=nothing,
                                         adaptive_loss=nothing,
                                         logger=nothing,
                                         log_options=LogOptions(),
                                         iteration=nothing,
                                         kwargs...) where iip
        if init_params === nothing
            if chain isa AbstractArray
                initθ = DiffEqFlux.initial_params.(chain)
            else
                initθ = DiffEqFlux.initial_params(chain)
            end
        else
            initθ = init_params
        end

        multioutput = typeof(chain) <: AbstractArray

        type_initθ = multioutput ? Base.promote_typeof.(initθ)[1] : Base.promote_typeof(initθ)

        if phi === nothing
            if multioutput
                _phi = Phi.(chain)
            else
                _phi = Phi(chain)
            end
        else
            _phi = phi
        end

        if derivative === nothing
            _derivative = numeric_derivative
        else
            _derivative = derivative
        end

        if !(typeof(adaptive_loss) <: AbstractAdaptiveLoss)
            floattype = eltype(initθ)
            if floattype <: Vector
                floattype = eltype(floattype)
            end
            adaptive_loss = NonAdaptiveLoss{floattype}()
        end

        if iteration isa Vector{Int64}
            self_increment = false
        else
            iteration = [1]
            self_increment = true
        end

        new{typeof(strategy),typeof(initθ),typeof(_phi),typeof(_derivative),typeof(param_estim),
            typeof(additional_loss),typeof(adaptive_loss),typeof(logger),typeof(kwargs)}(
            strategy,initθ,_phi,_derivative,param_estim,additional_loss,adaptive_loss,logger,
            log_options,iteration,self_increment,multioutput,kwargs)
    end
end

function vectorify(x, t::Type{T}) where T <: Real
    convertfunc(y) = convert(t, y)
    returnval = 
    if x isa Vector
        convertfunc.(x)
    else
        t[convertfunc(x)]
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
get_dict_vars(vars) = Dict( [Symbol(v) .=> i for (i,v) in enumerate(vars)])

# Wrapper for _transform_expression
function transform_expression(ex,indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput::Bool,eltypeθ,strategy,phi,derivative,integral,initθ;is_integral=false, dict_transformation_vars = nothing, transformation_vars = nothing)
    if ex isa Expr
        ex = _transform_expression(ex,indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput,eltypeθ,strategy,phi,derivative,integral,initθ;is_integral = is_integral,  dict_transformation_vars = dict_transformation_vars, transformation_vars = transformation_vars)
    end
    return ex
end

function get_ε(dim, der_num,eltypeθ)
    epsilon = cbrt(eps(eltypeθ))
    ε = zeros(eltypeθ, dim)
    ε[der_num] = epsilon
    ε
end

function get_limits(domain)
    if domain isa AbstractInterval
        return [leftendpoint(domain)], [rightendpoint(domain)]
    elseif domain isa ProductDomain
        return collect(map(leftendpoint , DomainSets.components(domain))), collect(map(rightendpoint , DomainSets.components(domain)))
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
function _transform_expression(ex,indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput::Bool,eltypeθ,strategy,phi,derivative_,integral,initθ;is_integral=false, dict_transformation_vars = nothing, transformation_vars = nothing)
    _args = ex.args
    for (i,e) in enumerate(_args)
        if !(e isa Expr)
            if e in keys(dict_depvars)
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                var_ = is_integral ? :(u) : :($(Expr(:$, :u)))
                ex.args = if !multioutput
                    [var_, Symbol(:cord, num_depvar), :($θ), :phi]
                else
                    [var_, Symbol(:cord, num_depvar), Symbol(:($θ), num_depvar), Symbol(:phi, num_depvar)]
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
                dict_interior_indvars = Dict([indvar .=> j for (j, indvar) in enumerate(dict_depvar_input[depvar])])                
                dim_l = length(dict_interior_indvars)

                var_ = is_integral ? :(derivative) : :($(Expr(:$, :derivative)))
                εs = [get_ε(dim_l, d, eltypeθ) for d in 1:dim_l]
                undv = [dict_interior_indvars[d_p] for d_p  in derivative_variables]
                εs_dnv = [εs[d] for d in undv]

                ex.args = if !multioutput
                    [var_, :phi, :u, Symbol(:cord, num_depvar), εs_dnv, order, :($θ)]
                else
                    [var_, Symbol(:phi, num_depvar), :u, Symbol(:cord, num_depvar), εs_dnv, order, Symbol(:($θ), num_depvar)]
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
                integrand_expr =_args[2]
                for d in depvars
                    d_ex = find_thing_in_expr(integrand_expr,d)
                    if !isempty(d_ex)
                        push!(integrating_depvars, d_ex[1].args[1])
                    end
                end

                lb, ub = get_limits(_args[1].domain.domain)
                lb, ub, _args[2], dict_transformation_vars, transformation_vars = transform_inf_integral(lb, ub, _args[2],integrating_depvars, dict_depvar_input, dict_depvars, integrating_variable, eltypeθ)            

                num_depvar = map(int_depvar -> dict_depvars[int_depvar], integrating_depvars)
                integrand_ = transform_expression(_args[2],indvars,depvars,dict_indvars,dict_depvars,
                                                dict_depvar_input, multioutput,eltypeθ,strategy,
                                                phi,derivative_,integral,initθ; is_integral = false, 
                                                dict_transformation_vars = dict_transformation_vars, 
                                                transformation_vars = transformation_vars)
                integrand__ = _dot_(integrand_)

                integrand = build_symbolic_loss_function(nothing, indvars,depvars,dict_indvars,dict_depvars,
                                                         dict_depvar_input, phi, derivative_, nothing, multioutput,
                                                         initθ, strategy, integrand = integrand__,
                                                         integrating_depvars=integrating_depvars,
                                                         eq_params=SciMLBase.NullParameters(),
                                                         dict_transformation_vars = dict_transformation_vars, 
                                                         transformation_vars = transformation_vars,
                                                         param_estim =false, default_p = nothing)
                # integrand = repr(integrand)
                lb = toexpr.(lb)
                ub = toexpr.(ub)
                ub_ = []
                lb_ = []
                for l in lb
                    if l isa Number
                        push!(lb_, l)
                    else
                        l_expr = NeuralPDE.build_symbolic_loss_function(nothing, indvars,depvars,
                                                                   dict_indvars,dict_depvars,
                                                                   dict_depvar_input, phi, derivative_,
                                                                   nothing, multioutput, initθ, strategy,
                                                                   integrand = _dot_(l), integrating_depvars=integrating_depvars,
                                                                   param_estim =false, default_p = nothing)
                        l_f = @RuntimeGeneratedFunction(l_expr)
                        push!(lb_, l_f)
                    end
                end
                for u_ in ub
                    if u_ isa Number
                        push!(ub_, u_)
                    else
                        u_expr = NeuralPDE.build_symbolic_loss_function(nothing, indvars,depvars,
                                                                    dict_indvars,dict_depvars,
                                                                    dict_depvar_input, phi, derivative_,
                                                                    nothing, multioutput, initθ, strategy,
                                                                    integrand = _dot_(u_), integrating_depvars=integrating_depvars,
                                                                    param_estim =false, default_p = nothing)
                        u_f = @RuntimeGeneratedFunction(u_expr)
                        push!(ub_, u_f)
                    end
                end

                integrand_func = @RuntimeGeneratedFunction(integrand)
                ex.args = [:($(Expr(:$, :integral))), :u, Symbol(:cord, num_depvar[1]), :phi, integrating_var_id, integrand_func, lb_, ub_,  :($θ)]
                break
            end
        else
            ex.args[i] = _transform_expression(ex.args[i],indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput,eltypeθ,strategy,phi,derivative_,integral,initθ; is_integral = is_integral, dict_transformation_vars = dict_transformation_vars, transformation_vars = transformation_vars)
        end
    end
    return ex
end

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

function build_symbolic_equation(eq,_indvars,_depvars,multioutput::Bool,eltypeθ,strategy,phi,derivative,initθ)
    depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    parse_equation(eq,indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput,eltypeθ,strategy,phi,derivative,integral,initθ)
end

function parse_equation(eq,indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput::Bool,eltypeθ,
                        strategy,phi,derivative,integral,initθ)
    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)
    left_expr = transform_expression(toexpr(eq_lhs),indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput,
                                     eltypeθ,strategy,phi,derivative,integral,initθ)
     right_expr = transform_expression(toexpr(eq_rhs),indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput,
                                       eltypeθ,strategy,phi,derivative,integral,initθ)
    left_expr = _dot_(left_expr)
    right_expr = _dot_(right_expr)
    loss_func = :($left_expr .- $right_expr)
end

"""
Build a loss function for a PDE or a boundary condition

# Examples: System of PDEs:

Take expressions in the form:

[Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
 Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

to

:((cord, θ, phi, derivative, u)->begin
          #= ... =#
          #= ... =#
          begin
              (θ1, θ2) = (θ[1:33], θ"[34:66])
              (phi1, phi2) = (phi[1], phi[2])
              let (x, y) = (cord[1], cord[2])
                  [(+)(derivative(phi1, u, [x, y], [[ε, 0.0]], 1, θ1), (*)(4, derivative(phi2, u, [x, y], [[0.0, ε]], 1, θ2))) - 0,
                   (+)(derivative(phi2, u, [x, y], [[ε, 0.0]], 1, θ2), (*)(9, derivative(phi1, u, [x, y], [[0.0, ε]], 1, θ1))) - 0]
              end
          end
      end)
"""
function build_symbolic_loss_function(eqs,_indvars,_depvars,dict_depvar_input,
                                      phi, derivative,integral,multioutput::Bool,initθ,strategy;
                                      bc_indvars=nothing,
                                      eq_params = SciMLBase.NullParameters(),
                                      param_estim = false,
                                      default_p=nothing,
                                      integrand=nothing,
                                      dict_transformation_vars = nothing, 
                                      transformation_vars = nothing,
                                      integrating_depvars=nothing)
    # dictionaries: variable -> unique number
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars == nothing ? indvars : bc_indvars
    integrating_depvars = integrating_depvars == nothing ? depvars : integrating_depvars
    return build_symbolic_loss_function(eqs,indvars,depvars,
                                        dict_indvars,dict_depvars,dict_depvar_input,
                                        phi,derivative,integral,multioutput,initθ,strategy;
                                        bc_indvars = bc_indvars,
                                        eq_params = eq_params,
                                        param_estim = param_estim,
                                        default_p=default_p,
                                        integrand=integrand,
                                        dict_transformation_vars = dict_transformation_vars, 
                                        transformation_vars = transformation_vars,
                                        integrating_depvars=integrating_depvars)
end

function get_indvars_ex(bc_indvars) # , dict_this_eq_indvars)
    i_=1
    indvars_ex = map(bc_indvars) do u
        if u isa Symbol
             # i = dict_this_eq_indvars[u]
             # ex = :($:cord[[$i],:])
             ex = :($:cord[[$i_],:])
             i_+=1
             ex
        else
           :(fill($u,size($:cord[[1],:])))
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
        if !isempty(find_thing_in_expr(expr,  depvar))
            dict_depvars[depvar] => dict_depvar_input[depvar]
        end
    end
    Dict(filter(p -> p !== nothing, pair_))
end

function build_symbolic_loss_function(eqs,indvars,depvars,
                                      dict_indvars,dict_depvars,dict_depvar_input,
                                      phi,derivative,integral,multioutput::Bool,initθ,strategy;
                                      eq_params = SciMLBase.NullParameters(),
                                      param_estim = param_estim,
                                      default_p=default_p,
                                      bc_indvars=indvars,
                                      integrand=nothing,
                                      dict_transformation_vars = nothing, 
                                      transformation_vars = nothing,
                                      integrating_depvars=depvars,
                                      )
    if multioutput
        eltypeθ = eltype(initθ[1])
    else
        eltypeθ = eltype(initθ)
    end

    if integrand isa Nothing
        loss_function = parse_equation(eqs,indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,multioutput,eltypeθ,strategy,phi,derivative,integral,initθ)
        this_eq_pair = pair(eqs, depvars, dict_depvars, dict_depvar_input)
        this_eq_indvars = unique(vcat(values(this_eq_pair)...))
    else 
        this_eq_pair = Dict(map(intvars -> dict_depvars[intvars] => dict_depvar_input[intvars], integrating_depvars))
        this_eq_indvars = transformation_vars isa Nothing ? unique(vcat(values(this_eq_pair)...)) : transformation_vars
        loss_function = integrand
    end
    vars = :(cord, $θ, phi, derivative, integral,u,p)
    ex = Expr(:block)
    if multioutput
        θ_nums = Symbol[]
        phi_nums = Symbol[]
        for v in depvars
            num = dict_depvars[v]
            push!(θ_nums,:($(Symbol(:($θ),num))))
            push!(phi_nums,:($(Symbol(:phi,num))))
        end

        expr_θ = Expr[]
        expr_phi = Expr[]

        acum =  [0;accumulate(+, length.(initθ))]
        sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]

        for i in eachindex(depvars)
            push!(expr_θ, :($θ[$(sep[i])]))
            push!(expr_phi, :(phi[$i]))
        end

        vars_θ = Expr(:(=), build_expr(:tuple, θ_nums), build_expr(:tuple, expr_θ))
        push!(ex.args,  vars_θ)

        vars_phi = Expr(:(=), build_expr(:tuple, phi_nums), build_expr(:tuple, expr_phi))
        push!(ex.args,  vars_phi)
    end
    #Add an expression for parameter symbols
    if param_estim == true && eq_params != SciMLBase.NullParameters()
        param_len = length(eq_params)
        last_indx =  [0;accumulate(+, length.(initθ))][end]
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i,eq_param) in enumerate(eq_params)
            push!(expr_params, :($θ[$(i+last_indx:i+last_indx)]))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols), build_expr(:tuple, expr_params))
        push!(ex.args,  params_eq)
    end

    if eq_params != SciMLBase.NullParameters() && param_estim == false
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i , eq_param) in enumerate(eq_params)
            push!(expr_params, :(ArrayInterface.allowed_getindex(p,$i:$i)))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols), build_expr(:tuple, expr_params))
        push!(ex.args,  params_eq)
    end

    eq_pair_expr = Expr[]
    for i in keys(this_eq_pair)
        push!(eq_pair_expr, :( $(Symbol(:cord, :($i))) = vcat($(this_eq_pair[i]...))))
    end
    vcat_expr = Expr(:block, :($(eq_pair_expr...)))
    vcat_expr_loss_functions = Expr(:block, vcat_expr, loss_function) # TODO rename

    if strategy isa QuadratureTraining
        indvars_ex = get_indvars_ex(bc_indvars)
        left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex 
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
    else
        indvars_ex = [:($:cord[[$i],:]) for (i, u) ∈ enumerate(this_eq_indvars)]
        left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
    end

    if !(dict_transformation_vars isa Nothing)
        transformation_expr_ = Expr[]

        for (i,u) in dict_transformation_vars
            push!(transformation_expr_, :($i = $u))
        end
        transformation_expr = Expr(:block, :($(transformation_expr_...)))
        vcat_expr_loss_functions = Expr(:block, transformation_expr, vcat_expr, loss_function)
    end

    let_ex = Expr(:let, vars_eq, vcat_expr_loss_functions)
    push!(ex.args,  let_ex)

    expr_loss_function = :(($vars) -> begin $ex end)
end

function build_loss_function(eqs,_indvars,_depvars,phi,derivative,integral,
                             multioutput::Bool,initθ,strategy;
                             bc_indvars=nothing,
                             eq_params=SciMLBase.NullParameters(),
                             param_estim=false,
                             default_p=nothing)
    # dictionaries: variable -> unique number
    depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars==nothing ? indvars : bc_indvars
    return build_loss_function(eqs,indvars,depvars,
                               dict_indvars,dict_depvars,dict_depvar_input,
                               phi,derivative,integral,multioutput,initθ,strategy;
                               bc_indvars=bc_indvars,
                               integration_indvars=indvars,
                               eq_params=eq_params,
                               param_estim=param_estim,
                               default_p=default_p)
end

function build_loss_function(eqs,indvars,depvars,
                             dict_indvars,dict_depvars,dict_depvar_input,
                             phi,derivative,integral,multioutput::Bool,initθ,strategy;
                             bc_indvars = indvars,
                             integration_indvars=indvars,
                             eq_params=SciMLBase.NullParameters(),
                             param_estim=false,
                             default_p=nothing)
     expr_loss_function = build_symbolic_loss_function(eqs,indvars,depvars,
                                                       dict_indvars,dict_depvars, dict_depvar_input,
                                                       phi,derivative,integral,multioutput,initθ,strategy;
                                                       bc_indvars = bc_indvars,
                                                       eq_params = eq_params,
                                                       param_estim=param_estim,default_p=default_p)
    u = get_u()
    _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
    loss_function = (cord, θ) -> begin
        _loss_function(cord, θ, phi, derivative, integral, u, default_p)
    end
    return loss_function
end

function get_vars(indvars_, depvars_)
    indvars = ModelingToolkit.getname.(indvars_)
    depvars = Symbol[]
    dict_depvar_input = Dict{Symbol,Vector{Symbol}}()
    for d in depvars_
        if ModelingToolkit.value(d) isa Term
            dname = ModelingToolkit.getname(d)
            push!(depvars, dname)
            push!(dict_depvar_input, dname => [nameof(ModelingToolkit.value(argument)) for argument in ModelingToolkit.value(d).arguments])
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
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    get_integration_variables(eqs, dict_indvars, dict_depvars)
end

function get_integration_variables(eqs, dict_indvars, dict_depvars)
    exprs = toexpr.(eqs)
    vars = map(exprs) do expr
        _vars =  Symbol.(filter(indvar -> length(find_thing_in_expr(expr,  indvar)) > 0, sort(collect(keys(dict_indvars)))))
    end
end

function get_variables(eqs, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    return get_variables(eqs, dict_indvars, dict_depvars)
end

function get_variables(eqs,dict_indvars,dict_depvars)
    bc_args = get_argument(eqs,dict_indvars,dict_depvars)
    return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
end

function get_number(eqs,dict_indvars,dict_depvars)
    bc_args = get_argument(eqs,dict_indvars,dict_depvars)
    return map(barg -> filter(x -> x isa Number, barg), bc_args)
end

function find_thing_in_expr(ex::Expr, thing; ans = [])
    if thing in ex.args
        push!(ans,ex)
    end
    for e in ex.args
        if e isa Expr
            if thing in e.args
                push!(ans,e)
            end
            find_thing_in_expr(e,thing; ans=ans)
        end
    end
    return collect(Set(ans))
end

# Get arguments from boundary condition functions
function get_argument(eqs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    get_argument(eqs,dict_indvars,dict_depvars)
end
function get_argument(eqs,dict_indvars,dict_depvars)
    exprs = toexpr.(eqs)
    vars = map(exprs) do expr
        _vars =  map(depvar -> find_thing_in_expr(expr,  depvar), collect(keys(dict_depvars)))
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


function generate_training_sets(domains,dx,eqs,bcs,eltypeθ,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    return generate_training_sets(domains,dx,eqs,bcs,eltypeθ,dict_indvars,dict_depvars)
end
# Generate training set in the domain and on the boundary
function generate_training_sets(domains,dx,eqs,bcs,eltypeθ,dict_indvars::Dict,dict_depvars::Dict)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx,length(domains))
    end

    spans = [infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,dxs)]
    dict_var_span = Dict([Symbol(d.variables) => infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,dxs)])

    bound_args = get_argument(bcs,dict_indvars,dict_depvars)
    bound_vars = get_variables(bcs,dict_indvars,dict_depvars)

    dif = [eltypeθ[] for i=1:size(domains)[1]]
    for _args in bound_args
        for (i,x) in enumerate(_args)
            if x isa Number
                push!(dif[i],x)
            end
        end
    end
    cord_train_set = collect.(spans)
    bc_data = map(zip(dif,cord_train_set)) do (d,c)
        setdiff(c, d)
    end

    dict_var_span_ = Dict([Symbol(d.variables) => bc for (d,bc) in zip(domains,bc_data)])

    bcs_train_sets = map(bound_args) do bt
        span = map(b -> get(dict_var_span, b, b), bt)
        _set = adapt(eltypeθ,hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end

    pde_vars = get_variables(eqs,dict_indvars,dict_depvars)
    pde_args = get_argument(eqs,dict_indvars,dict_depvars)

    pde_train_set = adapt(eltypeθ, hcat(vec(map(points -> collect(points), Iterators.product(bc_data...)))...))

    pde_train_sets = map(pde_args) do bt
        span = map(b -> get(dict_var_span_, b, b), bt)
        _set = adapt(eltypeθ,hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end
    [pde_train_sets,bcs_train_sets]
end

function get_bounds(domains,eqs,bcs,eltypeθ,_indvars::Array,_depvars::Array,strategy)
    depvars,indvars,dict_indvars,dict_depvars,dict_depvar_input = get_vars(_indvars, _depvars)
    return get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy)
end

function get_bounds(domains,eqs,bcs,eltypeθ,_indvars::Array,_depvars::Array,strategy::QuadratureTraining)
    depvars,indvars,dict_indvars,dict_depvars,dict_depvar_input = get_vars(_indvars, _depvars)
    return get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy)
end

function get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy::QuadratureTraining)
    dict_lower_bound = Dict([Symbol(d.variables) => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => supremum(d.domain) for d in domains])

    pde_args = get_argument(eqs,dict_indvars,dict_depvars)

    pde_lower_bounds= map(pde_args) do pd
        span = map(p -> get(dict_lower_bound, p, p), pd)
        map(s -> adapt(eltypeθ,s) + cbrt(eps(eltypeθ)), span)
    end
    pde_upper_bounds= map(pde_args) do pd
        span = map(p -> get(dict_upper_bound, p, p), pd)
        map(s -> adapt(eltypeθ,s) - cbrt(eps(eltypeθ)), span)
    end
    pde_bounds= [pde_lower_bounds,pde_upper_bounds]

    bound_vars = get_variables(bcs,dict_indvars,dict_depvars)

    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds= [bcs_lower_bounds,bcs_upper_bounds]

    [pde_bounds, bcs_bounds]
end

function get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy)
    dx = 1 / strategy.points
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain)+dx, supremum(d.domain)-dx] for d in domains])
    # pde_bounds = [[infimum(d.domain),supremum(d.domain)] for d in domains]
    pde_args = get_argument(eqs,dict_indvars,dict_depvars)

    pde_bounds= map(pde_args) do pd
        span = map(p -> get(dict_span, p, p), pd)
        map(s -> adapt(eltypeθ,s), span)
    end

    bound_args = get_argument(bcs,dict_indvars,dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)] for d in domains])

    bcs_bounds= map(bound_args) do bt
        span = map(b -> get(dict_span, b, b), bt)
        map(s -> adapt(eltypeθ,s), span)
    end
    [pde_bounds,bcs_bounds]
end

"""
An encoding of the test function phi that is used for calculating the PDE
value at domain points x

Fields:

- `f`: A representation of the chain function. If FastChain, then `f(x,p)`,
  if Chain then `f(p)(x)` (from Flux.destructure)
"""
struct Phi{C}
    f::C
    Phi(chain::FastChain) = new{typeof(chain)}(chain)
    Phi(chain::Flux.Chain) =  (re = Flux.destructure(chain)[2]; new{typeof(re)}(re))
end

(f::Phi{<:FastChain})(x,θ) = f.f(adapt(parameterless_type(θ),x),θ)
(f::Phi{<:Optimisers.Restructure})(x,θ) = f.f(θ)(adapt(parameterless_type(θ),x))

function get_u()
    u = (cord, θ, phi)-> phi(cord, θ)
end

# the method to calculate the derivative
function numeric_derivative(phi,u,x,εs,order,θ)
    _epsilon = one(eltype(θ)) / (2*cbrt(eps(eltype(θ))))
    ε = εs[order]
    ε = adapt(parameterless_type(θ),ε)
    x = adapt(parameterless_type(θ),x)
    if order > 1
        return (numeric_derivative(phi,u,x .+ ε,εs,order-1,θ)
              .- numeric_derivative(phi,u,x .- ε,εs,order-1,θ)) .* _epsilon
    else
        return (u(x .+ ε,θ,phi) .- u(x .- ε,θ,phi)) .* _epsilon
    end
end

function get_numeric_integral(strategy, _indvars, _depvars, multioutput::Bool, derivative)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    integral =
        (u, cord, phi, integrating_var_id, integrand_func, lb, ub, θ ;strategy=strategy, indvars=indvars, depvars=depvars, dict_indvars=dict_indvars, dict_depvars=dict_depvars)->
            begin
                
                function integration_(cord, lb, ub, θ)
                    cord_ = cord
                    function integrand_(x , p)
                        @Zygote.ignore @views(cord_[integrating_var_id]) .= x
                        return integrand_func(cord_, p, phi, derivative, nothing, u, nothing)
                    end
                    prob_ = IntegralProblem(integrand_,lb, ub ,θ)
                    sol = solve(prob_,CubatureJLh(),reltol=1e-3,abstol=1e-3)[1]

                    return sol
                end

                lb_ = zeros(size(lb)[1], size(cord)[2])
                ub_ = zeros(size(ub)[1], size(cord)[2])
                for (i, l) in enumerate(lb)
                    if l isa Number
                        @Zygote.ignore lb_[i, :] = fill(l, 1, size(cord)[2])
                    else
                        @Zygote.ignore lb_[i, :] = l(cord , θ, phi, derivative, nothing, u, nothing)
                    end
                end
                for (i, u_) in enumerate(ub)
                    if u_ isa Number
                        @Zygote.ignore ub_[i, :] = fill(u_, 1, size(cord)[2])
                    else
                        @Zygote.ignore ub_[i, :] = u_(cord , θ, phi, derivative, nothing, u, nothing)
                    end
                end
                integration_arr = Matrix{Float64}(undef, 1, 0)
                for i in 1:size(cord)[2]
                    # ub__ = @Zygote.ignore getindex(ub_, :,  i)
                    # lb__ = @Zygote.ignore getindex(lb_, :,  i)
                    integration_arr = hcat(integration_arr ,integration_(cord[:, i], lb_[:, i], ub_[:, i], θ))
                end
                return integration_arr
            end
end

function get_loss_function(loss_function, train_set, eltypeθ,strategy::GridTraining;τ=nothing)
    loss = (θ) -> mean(abs2,loss_function(train_set, θ))
end
            
@nograd function generate_random_points(points, bound, eltypeθ)
    function f(b)
      if b isa Number
           fill(eltypeθ(b),(1,points))
       else
           lb, ub =  b[1], b[2]
           lb .+ (ub .- lb) .* rand(eltypeθ,1,points)
       end
    end
    vcat(f.(bound)...)
end

function get_loss_function(loss_function, bound, eltypeθ, strategy::StochasticTraining;τ=nothing)
    points = strategy.points

    loss = (θ) -> begin
        sets = generate_random_points(points, bound,eltypeθ)
        sets_ = adapt(parameterless_type(θ),sets)
        mean(abs2,loss_function(sets_, θ))
    end
    return loss
end

@nograd function generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
    function f(b)
      if b isa Number
           fill(eltypeθ(b),(1,points))
       else
           lb, ub =  eltypeθ[b[1]], [b[2]]
           QuasiMonteCarlo.sample(points,lb,ub,sampling_alg)
       end
    end
    vcat(f.(bound)...)
end

function generate_quasi_random_points_batch(points, bound, eltypeθ, sampling_alg,minibatch)
    map(bound) do b
        if !(b isa Number)
            lb, ub =  [b[1]], [b[2]]
            set_ = QuasiMonteCarlo.generate_design_matrices(points,lb,ub,sampling_alg,minibatch)
            set = map(s -> adapt(eltypeθ,s), set_)
        else
            set = fill(eltypeθ(b),(1,points))
        end
    end
end

function get_loss_function(loss_function, bound, eltypeθ,strategy::QuasiRandomTraining;τ=nothing)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    resampling = strategy.resampling
    minibatch = strategy.minibatch

    point_batch = nothing
    point_batch = if resampling == false
        generate_quasi_random_points_batch(points, bound,eltypeθ,sampling_alg,minibatch)
    end
    loss =
        if resampling == true
            θ -> begin
                sets = generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
                sets_ = adapt(parameterless_type(θ),sets)
                mean(abs2,loss_function(sets_, θ))
            end
        else
            θ -> begin
                sets =  [point_batch[i] isa Array{eltypeθ,2} ?
                         point_batch[i] : point_batch[i][rand(1:minibatch)]
                                            for i in 1:length(point_batch)] #TODO
                sets_ = vcat(sets...)
                sets__ = adapt(parameterless_type(θ),sets_)
                mean(abs2,loss_function(sets__, θ))
            end
        end
    return loss
end

function get_loss_function(loss_function, lb,ub ,eltypeθ, strategy::QuadratureTraining;τ=nothing)

    if length(lb) == 0
        loss = (θ) -> mean(abs2,loss_function(rand(eltypeθ,1,10), θ))
        return loss
    end
    area = eltypeθ(prod(abs.(ub .-lb)))
    f_ = (lb,ub,loss_,θ) -> begin
        # last_x = 1
        function _loss(x,θ)
            # last_x = x
            # mean(abs2,loss_(x,θ), dims=2)
            # size_x = fill(size(x)[2],(1,1))
            x = adapt(parameterless_type(θ),x)
            sum(abs2,loss_(x,θ), dims=2) #./ size_x
        end
        prob = IntegralProblem(_loss,lb,ub,θ,batch = strategy.batch,nout=1)
        solve(prob,
              strategy.quadrature_alg,
              reltol = strategy.reltol,
              abstol = strategy.abstol,
              maxiters = strategy.maxiters)[1]
    end
    loss = (θ) -> 1/area* f_(lb,ub,loss_function,θ)
    return loss
end

function SciMLBase.symbolic_discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs
    
    domains = pde_system.domain
    eq_params = pde_system.ps
    defaults = pde_system.defaults
    default_p = eq_params == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in eq_params]

    param_estim = discretization.param_estim
    additional_loss = discretization.additional_loss

    # dimensionality of equation
    dim = length(domains)
    depvars,indvars,dict_indvars,dict_depvars,dict_depvar_input = get_vars(pde_system.indvars, pde_system.depvars)

    multioutput = discretization.multioutput
    initθ = discretization.init_params
    flat_initθ = multioutput ? reduce(vcat,initθ) : initθ
    eltypeθ = eltype(flat_initθ)
    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy
    integral = get_numeric_integral(strategy, pde_system.indvars, pde_system.depvars, multioutput, derivative)
    if !(eqs isa Array)
        eqs = [eqs]
    end

    pde_indvars = if strategy isa QuadratureTraining 
        get_argument(eqs,dict_indvars,dict_depvars)
    else
        get_variables(eqs,dict_indvars,dict_depvars)
    end
    pde_integration_vars = get_integration_variables(eqs,dict_indvars,dict_depvars)

    symbolic_pde_loss_functions = [build_symbolic_loss_function(eq,indvars,depvars,
                                                                dict_indvars,dict_depvars,dict_depvar_input,
                                                                phi, derivative,integral, multioutput,initθ,strategy;
                                                                eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                                                bc_indvars=pde_indvar
                                                                ) for (eq, pde_indvar) in zip(eqs, pde_indvars, pde_integration_vars)]

    bc_indvars = if strategy isa QuadratureTraining
         get_argument(bcs,dict_indvars,dict_depvars)
    else
         get_variables(bcs,dict_indvars,dict_depvars)
    end
    bc_integration_vars = get_integration_variables(bcs, dict_indvars, dict_depvars)
    symbolic_bc_loss_functions = [build_symbolic_loss_function(bc,indvars,depvars,
                                                               dict_indvars,dict_depvars, dict_depvar_input,
                                                               phi, derivative,integral,multioutput,initθ,strategy,
                                                               eq_params=eq_params,
                                                               param_estim=param_estim,
                                                               default_p=default_p;
                                                               bc_indvars=bc_indvar)
                                                               for (bc, bc_indvar) in zip(bcs, bc_indvars, bc_integration_vars)]
    symbolic_pde_loss_functions, symbolic_bc_loss_functions
end


function discretize_inner_functions(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs

    domains = pde_system.domain
    eq_params = pde_system.ps
    defaults = pde_system.defaults
    default_p = eq_params == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in eq_params]

    param_estim = discretization.param_estim
    additional_loss = discretization.additional_loss
    adaloss = discretization.adaptive_loss
    
    # dimensionality of equation
    dim = length(domains)

    #TODO fix it in MTK 6.0.0+v: ModelingToolkit.get_ivs(pde_system)
     depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = get_vars(pde_system.ivs,
                                                         pde_system.dvs)

    multioutput = discretization.multioutput
    initθ = discretization.init_params
    flat_initθ = multioutput ? reduce(vcat,initθ) : initθ
    eltypeθ = eltype(flat_initθ)

    flat_initθ = param_estim == false ? flat_initθ : vcat(flat_initθ, adapt(typeof(flat_initθ),default_p))
    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy
    integral = get_numeric_integral(strategy, pde_system.indvars, pde_system.depvars, multioutput, derivative)
    if !(eqs isa Array)
        eqs = [eqs]
    end
    pde_indvars = if strategy isa QuadratureTraining
        get_argument(eqs,dict_indvars,dict_depvars)
    else
        get_variables(eqs,dict_indvars,dict_depvars)
    end
   pde_integration_vars = get_integration_variables(eqs, dict_indvars, dict_depvars)
   _pde_loss_functions = [build_loss_function(eq,indvars,depvars,
                                             dict_indvars,dict_depvars,dict_depvar_input,
                                             phi, derivative,integral, multioutput, initθ,strategy;
                                             eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                             bc_indvars=pde_indvar, integration_indvars=integration_indvar
                                             ) for (eq, pde_indvar, integration_indvar) in zip(eqs, pde_indvars, pde_integration_vars)]
    bc_indvars = if strategy isa QuadratureTraining
         get_argument(bcs,dict_indvars,dict_depvars)
    else
         get_variables(bcs,dict_indvars,dict_depvars)
    end
    bc_integration_vars = get_integration_variables(bcs, dict_indvars, dict_depvars)

    _bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                              dict_indvars,dict_depvars, dict_depvar_input,
                                              phi,derivative,integral,multioutput,initθ,strategy;
                                              eq_params=eq_params,
                                              param_estim=param_estim,
                                              default_p=default_p,
                                              bc_indvars=bc_indvar,
                                              integration_indvars=integration_indvar) for (bc, bc_indvar, integration_indvar) in zip(bcs, bc_indvars, bc_integration_vars)]

    pde_loss_functions, bc_loss_functions =
    if strategy isa GridTraining
        dx = strategy.dx

        train_sets = generate_training_sets(domains,dx,eqs,bcs,eltypeθ,
                                            dict_indvars,dict_depvars)

        # the points in the domain and on the boundary
        pde_train_sets, bcs_train_sets = train_sets
        pde_train_sets = adapt.(parameterless_type(θ),pde_train_sets)
        bcs_train_sets =  adapt.(parameterless_type(θ),bcs_train_sets)
        pde_loss_functions = [get_loss_function(_loss,_set,eltypeθ,strategy)
                                                 for (_loss,_set) in zip(_pde_loss_functions,pde_train_sets)]

        bc_loss_functions =  [get_loss_function(_loss,_set,eltypeθ,strategy)
                                                 for (_loss,_set) in zip(_bc_loss_functions, bcs_train_sets)]
        (pde_loss_functions, bc_loss_functions)
    elseif strategy isa StochasticTraining
         bounds = get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy)
         pde_bounds, bcs_bounds = bounds

         pde_loss_functions = [get_loss_function(_loss,bound,eltypeθ,strategy)
                                                 for (_loss,bound) in zip(_pde_loss_functions, pde_bounds)]

          bc_loss_functions = [get_loss_function(_loss,bound,eltypeθ,strategy) 
                                                 for (_loss, bound) in zip(_bc_loss_functions, bcs_bounds)]
          (pde_loss_functions, bc_loss_functions)
    elseif strategy isa QuasiRandomTraining
         bounds = get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy)
         pde_bounds, bcs_bounds = bounds

         pde_loss_functions = [get_loss_function(_loss,bound,eltypeθ,strategy)
                                                  for (_loss,bound) in zip(_pde_loss_functions, pde_bounds)]

         strategy_ = QuasiRandomTraining(strategy.bcs_points;
                                         sampling_alg = strategy.sampling_alg,
                                         resampling = strategy.resampling,
                                         minibatch = strategy.minibatch)
         bc_loss_functions = [get_loss_function(_loss,bound,eltypeθ,strategy_)
                                                for (_loss, bound) in zip(_bc_loss_functions, bcs_bounds)]
         (pde_loss_functions, bc_loss_functions)
    elseif strategy isa QuadratureTraining
        bounds = get_bounds(domains,eqs,bcs,eltypeθ,dict_indvars,dict_depvars,strategy)
        pde_bounds, bcs_bounds = bounds

        lbs,ubs = pde_bounds
        pde_loss_functions = [get_loss_function(_loss,lb,ub,eltypeθ,strategy)
                                                 for (_loss,lb,ub) in zip(_pde_loss_functions, lbs,ubs )]
        lbs,ubs = bcs_bounds
        bc_loss_functions = [get_loss_function(_loss,lb,ub,eltypeθ,strategy)
                                                for (_loss,lb,ub) in zip(_bc_loss_functions, lbs,ubs)]

        (pde_loss_functions, bc_loss_functions)
    end

    # setup for all adaptive losses
    num_pde_losses = length(pde_loss_functions)
    num_bc_losses = length(bc_loss_functions)
    # assume one single additional loss function if there is one. this means that the user needs to lump all their functions into a single one,
    num_additional_loss = additional_loss isa Nothing ? 0 : 1 

    adaloss_T = eltype(adaloss.pde_loss_weights)
    logger = discretization.logger
    log_frequency = discretization.log_options.log_frequency
    iteration = discretization.iteration
    self_increment = discretization.self_increment

    # this will error if the user has provided a number of initial weights that is more than 1 and doesn't match the number of loss functions
    adaloss.pde_loss_weights = ones(adaloss_T, num_pde_losses) .* adaloss.pde_loss_weights
    adaloss.bc_loss_weights = ones(adaloss_T, num_bc_losses) .* adaloss.bc_loss_weights
    adaloss.additional_loss_weights = ones(adaloss_T, num_additional_loss) .* adaloss.additional_loss_weights


    # this is the function that gets called to do the adaptive reweighting, a function specific to the 
    # type of adaptive reweighting being performed. 
    # TODO: I'd love to pull this out and then dispatch on it via the AbstractAdaptiveLoss, so that users can implement their own
    #       currently this is kind of tricky since the different methods need different types of information, and the loss functions
    #       are generated internal to the code
    reweight_losses_func = 
    if adaloss isa GradientScaleAdaptiveLoss
        weight_change_inertia = discretization.adaptive_loss.weight_change_inertia
        function run_loss_gradients_adaptive_loss(θ)
            if iteration[1] % adaloss.reweight_every == 0
                # the paper assumes a single pde loss function, so here we grab the maximum of the maximums of each pde loss function
                pde_grads_maxes = [maximum(abs.(Zygote.gradient(pde_loss_function, θ)[1])) for pde_loss_function in pde_loss_functions]
                pde_grads_max = maximum(pde_grads_maxes)
                bc_grads_mean = [mean(abs.(Zygote.gradient(bc_loss_function, θ)[1])) for bc_loss_function in bc_loss_functions]

                nonzero_divisor_eps =  adaloss_T isa Float64 ? Float64(1e-11) : convert(adaloss_T, 1e-7)
                bc_loss_weights_proposed = pde_grads_max ./ (bc_grads_mean .+ nonzero_divisor_eps)
                adaloss.bc_loss_weights .= weight_change_inertia .* adaloss.bc_loss_weights .+ (1 .- weight_change_inertia) .* bc_loss_weights_proposed
                logscalar(logger, pde_grads_max, "adaptive_loss/pde_grad_max", iteration[1])
                logvector(logger, pde_grads_maxes, "adaptive_loss/pde_grad_maxes", iteration[1])
                logvector(logger, bc_grads_mean, "adaptive_loss/bc_grad_mean", iteration[1])
                logvector(logger, adaloss.bc_loss_weights, "adaptive_loss/bc_loss_weights", iteration[1])
            end
            nothing
        end
    elseif adaloss isa MiniMaxAdaptiveLoss
        pde_max_optimiser = adaloss.pde_max_optimiser
        bc_max_optimiser = adaloss.bc_max_optimiser
        function run_minimax_adaptive_loss(θ, pde_losses, bc_losses) 
            if iteration[1] % adaloss.reweight_every == 0
                Flux.Optimise.update!(pde_max_optimiser, adaloss.pde_loss_weights, -pde_losses)
                Flux.Optimise.update!(bc_max_optimiser, adaloss.bc_loss_weights, -bc_losses)
                logvector(logger, adaloss.pde_loss_weights, "adaptive_loss/pde_loss_weights", iteration[1])
                logvector(logger, adaloss.bc_loss_weights, "adaptive_loss/bc_loss_weights", iteration[1])
            end
            nothing
        end
    elseif adaloss isa NonAdaptiveLoss
        function run_nonadaptive_loss(θ)
            nothing
        end
    end

    function loss_function_(θ,p)

        # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
        pde_losses = [pde_loss_function(θ) for pde_loss_function in pde_loss_functions]
        bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

        # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
        # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
        Zygote.@ignore if self_increment 
            iteration[1] += 1
        end

        Zygote.@ignore begin
            if adaloss isa MiniMaxAdaptiveLoss
                reweight_losses_func(θ, pde_losses, bc_losses)
            else
                reweight_losses_func(θ)
            end
        end 
        weighted_pde_losses = adaloss.pde_loss_weights .* pde_losses
        weighted_bc_losses = adaloss.bc_loss_weights .* bc_losses

        sum_weighted_pde_losses = sum(weighted_pde_losses)
        sum_weighted_bc_losses = sum(weighted_bc_losses)
        weighted_loss_before_additional = sum_weighted_pde_losses + sum_weighted_bc_losses

        full_weighted_loss = 
        if additional_loss isa Nothing
            weighted_loss_before_additional
        else
            function _additional_loss(phi,θ)
                (θ_,p_) = if (param_estim == true)
                    θ[1:end - length(default_p)], θ[(end - length(default_p) + 1):end]
                else
                    θ, nothing
                end
                return additional_loss(phi, θ, p_)
            end
            weighted_additional_loss_val = adaloss.additional_loss_weights[1] * _additional_loss(phi, θ)
            weighted_loss_before_additional + weighted_additional_loss_val
        end

        Zygote.@ignore begin
            if iteration[1] % log_frequency == 0
                logvector(logger, pde_losses, "unweighted_loss/pde_losses", iteration[1])
                logvector(logger, bc_losses, "unweighted_loss/bc_losses", iteration[1])
                logvector(logger, weighted_pde_losses, "weighted_loss/weighted_pde_losses", iteration[1])
                logvector(logger, weighted_bc_losses, "weighted_loss/weighted_bc_losses", iteration[1])
                logscalar(logger, sum_weighted_pde_losses, "weighted_loss/sum_weighted_pde_losses", iteration[1])
                logscalar(logger, sum_weighted_bc_losses, "weighted_loss/sum_weighted_bc_losses", iteration[1])
                logscalar(logger, full_weighted_loss, "weighted_loss/full_weighted_loss", iteration[1])
                logvector(logger, adaloss.pde_loss_weights, "adaptive_loss/pde_loss_weights", iteration[1])
                logvector(logger, adaloss.bc_loss_weights, "adaptive_loss/bc_loss_weights", iteration[1])
            end
        end

        return full_weighted_loss
    end

    (bc_loss_functions=bc_loss_functions, pde_loss_functions=pde_loss_functions, full_loss_function=loss_function_, 
        additional_loss_function=additional_loss, flat_initθ=flat_initθ, 
        inner_pde_loss_functions=_pde_loss_functions, inner_bc_loss_functions=_bc_loss_functions)
end

# Convert a PDE problem into an OptimizationProblem
function SciMLBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    discretized_functions = discretize_inner_functions(pde_system, discretization)
    f = OptimizationFunction(discretized_functions.full_loss_function, Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, discretized_functions.flat_initθ)
end
