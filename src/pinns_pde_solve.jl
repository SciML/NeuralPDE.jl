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
Algorithm for solving Physics-Informed Neural Networks problems.

Arguments:
* `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `strategy`: determines which training strategy will be used,
* `init_params`: the initial parameter of the neural network,
* `phi`: a trial solution,
* `derivative`: method that calculates the derivative.

"""
abstract type AbstractPINN{isinplace} <: SciMLBase.SciMLProblem end

struct PhysicsInformedNN{isinplace,C,T,P,PH,DER,PE,AL,K} <: AbstractPINN{isinplace}
    chain::C
    strategy::T
    init_params::P
    phi::PH
    derivative::DER
    param_estim::PE
    additional_loss::AL
    kwargs::K

    @add_kwonly function PhysicsInformedNN{iip}(chain,
                                             strategy;
                                             init_params=nothing,
                                             phi=nothing,
                                             derivative=nothing,
                                             param_estim=false,
                                             additional_loss=nothing,
                                             kwargs...) where iip
        if init_params == nothing
            if chain isa AbstractArray
                initθ = DiffEqFlux.initial_params.(chain)
            else
                initθ = DiffEqFlux.initial_params(chain)
            end

        else
            initθ = init_params
        end
        type_initθ = if (typeof(chain) <: AbstractVector) Base.promote_typeof.(initθ)[1] else  Base.promote_typeof(initθ) end
        parameterless_type_θ = DiffEqBase.parameterless_type(type_initθ)

        if phi == nothing
            if chain isa AbstractArray
                _phi = get_phi.(chain, parameterless_type_θ)
            else
                _phi = get_phi(chain, parameterless_type_θ)
            end
        else
            _phi = phi
        end

        if derivative == nothing
            _derivative = get_numeric_derivative()
        else
            _derivative = derivative
        end
        new{iip,typeof(chain),typeof(strategy),typeof(initθ),typeof(_phi),typeof(_derivative),typeof(param_estim),typeof(additional_loss),typeof(kwargs)}(chain, strategy, initθ, _phi, _derivative, param_estim, additional_loss, kwargs)
    end
end
PhysicsInformedNN(chain,strategy,args...;kwargs...) = PhysicsInformedNN{true}(chain, strategy, args...;kwargs...)

SciMLBase.isinplace(prob::PhysicsInformedNN{iip}) where iip = iip


abstract type TrainingStrategies  end

"""
* `dx`: the discretization of the grid.
"""
struct GridTraining <: TrainingStrategies
    dx
end

"""
* `points`: number of points in random select training set,
* `bcs_points`: number of points in random select training set for boundry conditions (by default, it equals `points`).
"""
struct StochasticTraining <: TrainingStrategies
    points::Int64
    bcs_points::Int64
end
function StochasticTraining(points;bcs_points=points)
    StochasticTraining(points, bcs_points)
end
"""
* `points`:  the number of quasi-random points in a sample,
* `bcs_points`: the number of quasi-random points in a sample for boundry conditions (by default, it equals `points`),
* `sampling_alg`: the quasi-Monte Carlo sampling algorithm,
* `resampling`: if it's false - the full training set is generated in advance before training,
   and at each iteration, one subset is randomly selected out of the batch.
   if it's true - the training set isn't generated beforehand, and one set of quasi-random
   points is generated directly at each iteration in runtime. In this case `minibatch` has no effect,
* `minibatch`: the number of subsets, if resampling == false.

For more information look: QuasiMonteCarlo.jl https://github.com/SciML/QuasiMonteCarlo.jl
"""
struct QuasiRandomTraining <: TrainingStrategies
    points::Int64
    bcs_points::Int64
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
    resampling::Bool
    minibatch::Int64
end
function QuasiRandomTraining(points;bcs_points=points, sampling_alg=LatinHypercubeSample(),resampling=true, minibatch=0)
    QuasiRandomTraining(points, bcs_points, sampling_alg, resampling, minibatch)
end
"""
* `quadrature_alg`: quadrature algorithm,
* `reltol`: relative tolerance,
* `abstol`: absolute tolerance,
* `maxiters`: the maximum number of iterations in quadrature algorithm,
* `batch`: the preferred number of points to batch.

For more information look: Quadrature.jl https://github.com/SciML/Quadrature.jl
"""
struct QuadratureTraining <: TrainingStrategies
    quadrature_alg::DiffEqBase.AbstractQuadratureAlgorithm
    reltol::Float64
    abstol::Float64
    maxiters::Int64
    batch::Int64
end

function QuadratureTraining(;quadrature_alg=CubatureJLh(),reltol=1e-6,abstol=1e-3,maxiters=1e3,batch=100)
    QuadratureTraining(quadrature_alg, reltol, abstol, maxiters, batch)
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
function transform_expression(ex, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
    if ex isa Expr
        ex = _transform_expression(ex, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
    end
    return ex
end

function get_ε(dim, der_num, eltypeθ)
    epsilon = cbrt(eps(eltypeθ))
    ε = zeros(eltypeθ, dim)
    ε[der_num] = epsilon
    ε
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

function _transform_expression(ex, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
    _args = ex.args
    for (i, e) in enumerate(_args)
        if e isa Function && !(e isa ModelingToolkit.Differential)
            ex.args[i] = Symbol(e)
        end
        if !(e isa Expr)
            if e in keys(dict_depvars)
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                cord = :cord
                # map all the indvars
                interior_u_args = map(_args[2:end]) do arg
                    if arg isa Symbol
                        arg
                    else
                        :(fill($arg, (1, $:batch_size)))
                        @show :(fill($arg, (1, $:batch_size)))
                    end
                end
                @show interior_u_args
                input_cord = :(adapt(DiffEqBase.parameterless_type($θ), vcat($(interior_u_args...))))
                @show input_cord
                ex.args = if !(typeof(chain) <: AbstractVector)
                    [:($(Expr(:$, :u))), cord, :($θ), :phi]
                else
                    [:($(Expr(:$, :u))), cord, Symbol(:($θ),num_depvar), Symbol(:phi,num_depvar)]
                end
                break
            elseif e isa ModelingToolkit.Differential
                @info "differential"
                @show _args
                @show e
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
                cord = :cord
                dim_l = length(indvars)
                εs = [get_ε(dim_l, d, eltypeθ) for d in 1:dim_l]
                @show depvar
                @show dict_depvar_input
                dict_interior_indvars = Dict([indvar .=> j for (j, indvar) in enumerate(dict_depvar_input[depvar])])
                undv = [dict_interior_indvars[d_p] for d_p  in derivative_variables]
                εs_dnv = [εs[d] for d in undv]
                interior_u_args = map(_args[2:end]) do arg
                    if arg isa Symbol
                        arg
                    else
                        :(fill($arg, (1, $:batch_size)))
                        @show :(fill($arg, (1, $:batch_size)))
                    end
                end
                input_cord = :(adapt(DiffEqBase.parameterless_type($θ), vcat($(interior_u_args...))))
                ex.args = if !(typeof(chain) <: AbstractVector)
                    [:($(Expr(:$, :derivative))), :phi, :u, cord, εs_dnv, order, :($θ)]
                else
                    [:($(Expr(:$, :derivative))), Symbol(:phi,num_depvar), :u, cord, εs_dnv, order, Symbol(:($θ),num_depvar)]
                end
                break
            end
        else
            ex.args[i] = _transform_expression(ex.args[i], dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
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

function build_symbolic_equation(eq, _indvars, _depvars, chain, eltypeθ, strategy)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    parse_equation(eq, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
end

function parse_equation(eq, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)

    left_expr = transform_expression(toexpr(eq_lhs), dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
    right_expr = transform_expression(toexpr(eq_rhs), dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
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
function build_symbolic_loss_function(eqs,_indvars,_depvars, dict_depvar_input,
                                      phi, derivative,chain,initθ,strategy;
                                      bc_indvars=nothing,
                                      eq_params=SciMLBase.NullParameters(),
                                      param_estim=false,
                                      default_p=nothing,
                                      integration_indvars=nothing)
    # dictionaries: variable -> unique number
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars == nothing ? indvars : bc_indvars
    integration_indvars = integration_indvars == nothing ? indvars : integration_indvars
    return build_symbolic_loss_function(eqs,indvars,depvars,
                                        dict_indvars,dict_depvars,dict_depvar_input,
                                        phi,derivative,chain,initθ,strategy;
                                        bc_indvars=bc_indvars,
                                        eq_params=eq_params,
                                        param_estim=param_estim,
                                        default_p=default_p,
                                        integration_indvars=integration_indvars)
end

function get_indvars_ex(bc_indvars)
    i_ = 1
    indvars_ex = map(bc_indvars) do u
        if u isa Symbol
             ex = :($:cord[[$i_],:])
             i_ += 1
             ex
        else
           :(fill($u, size($:cord[[1],:])))
        end
    end
    indvars_ex
end

function build_symbolic_loss_function(eqs,indvars,depvars, 
                                      dict_indvars,dict_depvars,dict_depvar_input,
                                      phi,derivative,chain,initθ,strategy;
                                      eq_params=SciMLBase.NullParameters(),
                                      param_estim=param_estim,
                                      default_p=default_p,
                                      bc_indvars=indvars,
                                      integration_indvars=indvars)
    if chain isa AbstractArray
        eltypeθ = eltype(initθ[1])
    else
        eltypeθ = eltype(initθ)
    end

    loss_function = parse_equation(eqs, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)
    vars = :(cord, $θ, phi, derivative, u, p)
    ex = Expr(:block)
    if typeof(chain) <: AbstractVector
        θ_nums = Symbol[]
        phi_nums = Symbol[]
        for v in depvars
            num = dict_depvars[v]
            push!(θ_nums, :($(Symbol(:($θ), num))))
            push!(phi_nums, :($(Symbol(:phi, num))))
        end

        expr_θ = Expr[]
        expr_phi = Expr[]

        acum =  [0;accumulate(+, length.(initθ))]
        sep = [acum[i] + 1:acum[i + 1] for i in 1:length(acum) - 1]

        for i in eachindex(depvars)
            push!(expr_θ, :($θ[$(sep[i])]))
            push!(expr_phi, :(phi[$i]))
        end
        
        vars_θ = Expr(:(=), build_expr(:tuple, θ_nums), build_expr(:tuple, expr_θ))
        push!(ex.args,  vars_θ)

        vars_phi = Expr(:(=), build_expr(:tuple, phi_nums), build_expr(:tuple, expr_phi))
        push!(ex.args,  vars_phi)
    end
    # Add an expression for parameter symbols
    if param_estim == true && eq_params != SciMLBase.NullParameters()
        param_len = length(eq_params)
        last_indx =  [0;accumulate(+, length.(initθ))][end]
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i, eq_param) in enumerate(eq_params)
        push!(expr_params, :($θ[$(i + last_indx:i + last_indx)]))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols), build_expr(:tuple, expr_params))
        push!(ex.args,  params_eq)
    end

    if eq_params != SciMLBase.NullParameters() && param_estim == false
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i, eq_param) in enumerate(eq_params)
            push!(expr_params, :(ArrayInterface.allowed_getindex(p, $i:$i)))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols), build_expr(:tuple, expr_params))
        push!(ex.args,  params_eq)
    end

    #= 
    if strategy isa QuadratureTraining

        indvars_ex = get_indvars_ex(bc_indvars)

        left_arg_pairs, right_arg_pairs = indvars, indvars_ex
        vcat_expr =  :(cord = vcat($(indvars...)))
        vcat_expr_loss_functions = Expr(:block, vcat_expr, loss_function) # TODO rename
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))

    else
        indvars_ex = [:($:cord[[$i],:]) for (i, u) ∈ enumerate(indvars)]
        left_arg_pairs, right_arg_pairs = indvars, indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
        vcat_expr_loss_functions = loss_function # TODO rename
    end =#

    indvars_ex = [:($:cord[[$i],:]) for (i, u) ∈ enumerate(integration_indvars)]
    batch_size_variable = if length(integration_indvars) > 0
        :(size($:cord[[1], :])[2])
    else
        :(1)
    end
    left_arg_pairs, right_arg_pairs = [integration_indvars; :($:batch_size)], [indvars_ex; batch_size_variable]
    @info "in build_symbolic_loss_function"
    vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
    vcat_expr_loss_functions = loss_function # TODO rename
    
    @show indvars_ex
    @show left_arg_pairs, right_arg_pairs
    @show vars_eq
    @show vcat_expr_loss_functions

    let_ex = Expr(:let, vars_eq, vcat_expr_loss_functions)
    push!(ex.args,  let_ex)

    expr_loss_function = :(($vars) -> begin $ex end)
end


######### NORMAL EXAMPLE
@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 2D PDE
eq  = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

# Initial and boundary conditions
bcs = [u(0, x) ~ -sin(pi * x),
       u(t, -1) ~ 0.,
       u(t, 1) ~ 0.,
       u(t, -1) ~ u(t, 1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(-1.0, 1.0)]
# Discretization
dx = 0.05
# Neural network
chain = FastChain(FastDense(2, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
eltypeθ = eltype(initθ)
parameterless_type_θ = DiffEqBase.parameterless_type(initθ)
strategy = NeuralPDE.GridTraining(dx)

phi = NeuralPDE.get_phi(chain, parameterless_type_θ)
derivative = NeuralPDE.get_numeric_derivative()


_indvars = [t,x]
_depvars = [u]

depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)

Base.Broadcast.broadcasted(::typeof(transform_expression(eq, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)), cord, θ, phi) = transform_expression(eq, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)(cord, θ, phi)

_pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars, dict_depvar_input,
                                                   phi,derivative,chain,initθ,strategy)

bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,dict_depvar_input,
                                                    phi,derivative,chain,initθ,strategy,
                                                    bc_indvars=bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

train_sets = NeuralPDE.generate_training_sets(domains, dx, [eq], bcs, eltypeθ, indvars, depvars)
train_domain_set, train_bound_set = train_sets


pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                                                train_domain_set[1],
                                                eltypeθ,parameterless_type_θ,
                                                strategy)

bc_loss_functions = [NeuralPDE.get_loss_function(loss,set,
                                                 eltypeθ, parameterless_type_θ,
                                                 strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)]


loss_functions = [pde_loss_function; bc_loss_functions]
loss_function__ = θ -> sum(map(l -> l(θ), loss_functions))

function loss_function_(θ, p)
    return loss_function__(θ)
end

f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

cb_ = function (p, l)
    println("loss: ", l, "losses: ", map(l -> l(p), loss_functions))
    return false
end

# optimizer
opt = BFGS()
res = GalacticOptim.solve(prob, opt; cb=cb_, maxiters=2) 


function build_loss_function(eqs,_indvars,_depvars, dict_depvar_input, phi,derivative,
                             chain,initθ,strategy;
                             bc_indvars=nothing,
                             eq_params=SciMLBase.NullParameters(),
                             param_estim=false,
                             default_p=nothing)
    # dictionaries: variable -> unique number
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars == nothing ? indvars : bc_indvars
    return build_loss_function(eqs,indvars,depvars,
                               dict_indvars,dict_depvars, dict_depvar_input,
                               phi,derivative,chain,initθ,strategy;
                               bc_indvars=bc_indvars,
                               integration_indvars=indvars,
                               eq_params=eq_params,
                               param_estim=param_estim,
                               default_p=default_p)
end

function build_loss_function(eqs,indvars,depvars,
                             dict_indvars,dict_depvars, dict_depvar_input,
                             phi,derivative,chain,initθ,strategy;
                             bc_indvars=indvars,
                             integration_indvars=indvars,
                             eq_params=SciMLBase.NullParameters(),
                             param_estim=false,
                             default_p=nothing)
     expr_loss_function = build_symbolic_loss_function(eqs,indvars,depvars,
                                                       dict_indvars,dict_depvars, dict_depvar_input, 
                                                       phi,derivative,chain,initθ,strategy;
                                                       bc_indvars=bc_indvars,integration_indvars=indvars,
                                                       eq_params=eq_params,
                                                       param_estim=param_estim,default_p=default_p)
    u = get_u()
    _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
    loss_function = (cord, θ) -> begin
        _loss_function(cord, θ, phi, derivative, u, default_p)
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

function get_variables(eqs, dict_indvars, dict_depvars)
bc_args = get_argument(eqs, dict_indvars, dict_depvars)
    return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
end

function get_number(eqs, dict_indvars, dict_depvars)
bc_args = get_argument(eqs, dict_indvars, dict_depvars)
    return map(barg -> filter(x -> x isa Number, barg), bc_args)
end

function find_thing_in_expr(ex::Expr, thing; ans=Expr[])
    for e in ex.args
        if e isa Expr
            if thing in e.args
                push!(ans, e)
            end
            find_thing_in_expr(e, thing; ans=ans)
        end
    end
    return collect(Set(ans))
end

# Get arguments from boundary condition functions
function get_argument(eqs, _indvars::Array, _depvars::Array)
depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    get_argument(eqs, dict_indvars, dict_depvars)
end
function get_argument(eqs, dict_indvars, dict_depvars)
        exprs = toexpr.(eqs)
        vars = map(exprs) do expr
        _vars =  map(depvar -> find_thing_in_expr(expr,  depvar), collect(keys(dict_depvars)))
        f_vars = filter(x -> !isempty(x), _vars)
        map(x -> first(x), f_vars)
    end
    args_ = map(vars) do _vars
    ind_args_ = map(var -> var.args[2:end], _vars)
        unionarg_ = union(ind_args_...)        
    end
    return args_ # TODO for all arguments
end


function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    return generate_training_sets(domains, dx, eqs, bcs, eltypeθ, dict_indvars, dict_depvars)
end
# Generate training set in the domain and on the boundary
function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, dict_indvars::Dict, dict_depvars::Dict)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx, length(domains))
    end

    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    dict_var_span = Dict([Symbol(d.variables) => infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)])
    
    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    bound_vars = get_variables(bcs, dict_indvars, dict_depvars)
                
                dif = [eltypeθ[] for i = 1:size(domains)[1]]
    for _args in bound_args
        for (i, x) in enumerate(_args)
    if x isa Number
                push!(dif[i], x)
            end
        end
    end
    cord_train_set = collect.(spans)
    bc_data = map(zip(dif, cord_train_set)) do (d, c)
        setdiff(c, d)
        end

    dict_var_span_ = Dict([Symbol(d.variables) => bc for (d, bc) in zip(domains, bc_data)])

    bcs_train_sets = map(bound_args) do bt
        span = map(b -> get(dict_var_span, b, b), bt)
        _set = adapt(eltypeθ, hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end

    pde_vars = get_variables(eqs, dict_indvars, dict_depvars)
    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    pde_train_set = adapt(eltypeθ, hcat(vec(map(points -> collect(points), Iterators.product(bc_data...)))...))

    pde_train_sets = map(pde_args) do bt
        span = map(b -> get(dict_var_span_, b, b), bt)
        _set = adapt(eltypeθ, hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end
    [pde_train_sets,bcs_train_sets]
end

function get_bounds(domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array, strategy)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array, strategy::QuadratureTraining)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy::QuadratureTraining)
    dict_lower_bound = Dict([Symbol(d.variables) => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => supremum(d.domain) for d in domains])

    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    pde_lower_bounds = map(pde_args) do pd
    span = map(p -> get(dict_lower_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s) + cbrt(eps(eltypeθ)), span)
    end
    pde_upper_bounds = map(pde_args) do pd
    span = map(p -> get(dict_upper_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s) - cbrt(eps(eltypeθ)), span)
    end
    @info "in get_bounds by pde_bounds"
    @show pde_upper_bounds
    @show pde_lower_bounds
    pde_bounds = [pde_lower_bounds,pde_upper_bounds]

   
    bound_vars = get_variables(bcs, dict_indvars, dict_depvars)
        
    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds = [bcs_lower_bounds,bcs_upper_bounds]
    
    [pde_bounds, bcs_bounds]
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    dx = 1 / strategy.points
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain) + dx, supremum(d.domain) - dx] for d in domains])
    # pde_bounds = [[infimum(d.domain),supremum(d.domain)] for d in domains]
    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    pde_bounds = map(pde_args) do pd
        span = map(p -> get(dict_span, p, p), pd)
        map(s -> adapt(eltypeθ, s), span)
    end

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)] for d in domains])

    bcs_bounds = map(bound_args) do bt
span = map(b -> get(dict_span, b, b), bt)
        map(s -> adapt(eltypeθ, s), span)
    end
    [pde_bounds,bcs_bounds]
end

function get_phi(chain, parameterless_type_θ)
    # The phi trial solution
        if chain isa FastChain
        phi = (x, θ) -> chain(adapt(parameterless_type_θ, x), θ)
    else
        _, re  = Flux.destructure(chain)
        phi = (x, θ) -> re(θ)(adapt(parameterless_type_θ, x))
    end
    phi
end

function get_u()
    u = (cord, θ, phi) -> phi(cord, θ)
end

u = get_u()
# Base.Broadcast.broadcasted(::typeof(get_u()), cord, θ, phi) = get_u()(cord, θ, phi)

# the method to calculate the derivative
function get_numeric_derivative()
    derivative =
        (phi, u, x, εs, order, θ) ->
        begin
            _epsilon = one(eltype(θ)) / (2 * cbrt(eps(eltype(θ))))
            ε = εs[order]
            ε = adapt(DiffEqBase.parameterless_type(θ), ε)
            x = adapt(DiffEqBase.parameterless_type(θ), x)
            if order > 1
        return (derivative(phi, u, x .+ ε, εs, order - 1, θ)
                      .- derivative(phi, u, x .- ε, εs, order - 1, θ)) .* _epsilon
            else
                @show length(x)
                @show length(ε)
@show length(θ)
                return (u(x .+ ε, θ, phi) .- u(x .- ε, θ, phi)) .* _epsilon
            end
        end
end

derivative = get_numeric_derivative()
# Base.Broadcast.broadcasted(::typeof(get_numeric_derivative()), phi,u,x,εs,order,θ) = get_numeric_derivative()(phi,u,x,εs,order,θ)

    function get_loss_function(loss_function, train_set, eltypeθ, parameterless_type_θ, strategy::GridTraining;τ=nothing)
    @info "in get_loss_function"
    loss = (θ) -> mean(abs2, loss_function(train_set, θ))
end
            
@nograd function generate_random_points(points, bound, eltypeθ)
            function f(b)
      if b isa Number
           fill(eltypeθ(b), (1, points))
else
lb, ub =  b[1], b[2]
           lb .+ (ub .- lb) .* rand(eltypeθ, 1, points)
       end
    end
    vcat(f.(bound)...)
end

function get_loss_function(loss_function, bound, eltypeθ, parameterless_type_θ, strategy::StochasticTraining;τ=nothing)
    points = strategy.points

    loss = (θ) -> begin
        sets = generate_random_points(points, bound, eltypeθ)
        sets_ = adapt(parameterless_type_θ, sets)
        mean(abs2, loss_function(sets_, θ))
    end
    return loss
end

@nograd function generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
    function f(b)
      if b isa Number
           fill(eltypeθ(b), (1, points))
       else
           lb, ub =  eltypeθ[b[1]], [b[2]]
           QuasiMonteCarlo.sample(points, lb, ub, sampling_alg)
end
    end
    vcat(f.(bound)...)
end

function generate_quasi_random_points_batch(points, bound, eltypeθ, sampling_alg, minibatch)
    map(bound) do b
        if !(b isa Number)
            lb, ub =  [b[1]], [b[2]]
            set_ = QuasiMonteCarlo.generate_design_matrices(points, lb, ub, sampling_alg, minibatch)
            set = map(s -> adapt(eltypeθ, s), set_)
        else
            set = fill(eltypeθ(b), (1, points))
        end
    end
end

function get_loss_function(loss_function, bound, eltypeθ, parameterless_type_θ, strategy::QuasiRandomTraining;τ=nothing)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    resampling = strategy.resampling
    minibatch = strategy.minibatch

    point_batch = nothing
    point_batch = if resampling == false
        generate_quasi_random_points_batch(points, bound, eltypeθ, sampling_alg, minibatch)
    end
    loss =
        if resampling == true
            θ -> begin
            sets = generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
            sets_ = adapt(parameterless_type_θ, sets)
                mean(abs2, loss_function(sets_, θ))
    end
        else
            θ -> begin
                sets =  [point_batch[i] isa Array{eltypeθ,2} ?
                         point_batch[i] : point_batch[i][rand(1:minibatch)]
                                            for i in 1:length(point_batch)] # TODO
                sets_ = vcat(sets...)
                sets__ = adapt(parameterless_type_θ, sets_)
                mean(abs2, loss_function(sets__, θ))
            end
        end
    return loss
end

function get_loss_function(loss_function, lb, ub, eltypeθ, parameterless_type_θ, strategy::QuadratureTraining;τ=nothing)
    @info "in get_loss_function"
    if length(lb) == 0
        loss = (θ) -> mean(abs2, loss_function(rand(eltypeθ, 1, 10), θ))
        return loss
    end
    area = eltypeθ(prod(abs.(ub .- lb)))
    f_ = (lb, ub, loss_, θ) -> begin
        # last_x = 1
        function _loss(x, θ)
            # last_x = x
            # mean(abs2,loss_(x,θ), dims=2)
            # size_x = fill(size(x)[2],(1,1))
            x = adapt(parameterless_type_θ, x)
            sum(abs2, loss_(x, θ), dims=2) # ./ size_x
        end
        prob = QuadratureProblem(_loss, lb, ub, θ, batch=strategy.batch, nout=1)
        solve(prob,
              strategy.quadrature_alg,
              reltol=strategy.reltol,
              abstol=strategy.abstol,
              maxiters=strategy.maxiters)[1]
    end
    
    @show f_
    @show area
    loss = (θ) -> 1 / area * f_(lb, ub, loss_function, θ)
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
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(pde_system.indvars, pde_system.depvars)

    chain = discretization.chain
        initθ = discretization.init_params
    flat_initθ = if (typeof(chain) <: AbstractVector) reduce(vcat, initθ) else initθ end
    eltypeθ = eltype(flat_initθ)
    parameterless_type_θ =  DiffEqBase.parameterless_type(flat_initθ)
    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy
    if !(eqs isa Array)
        eqs = [eqs]
    end
    pde_indvars = if strategy isa QuadratureTraining 
        get_argument(eqs, dict_indvars, dict_depvars)
    else
        get_variables(eqs, dict_indvars, dict_depvars)
    end
    pde_integration_vars = get_integration_variables(eqs, dict_indvars, dict_depvars)
    @info "in symbolic discretize"
    @show dict_depvar_input
    symbolic_pde_loss_functions = [build_symbolic_loss_function(eq,indvars,depvars,
                                                                dict_indvars,dict_depvars,dict_depvar_input,
                                                                phi, derivative,chain,initθ,strategy;eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                                                bc_indvars=pde_indvar, integration_indvars=integration_indvar
                                                                ) for (eq, pde_indvar, integration_indvar) in zip(eqs, pde_indvars, pde_integration_vars)]

    bc_indvars = if strategy isa QuadratureTraining
         get_argument(bcs, dict_indvars, dict_depvars)
    else
         get_variables(bcs, dict_indvars, dict_depvars)
    end
    bc_integration_vars = get_integration_variables(bcs, dict_indvars, dict_depvars)
    symbolic_bc_loss_functions = [build_symbolic_loss_function(bc,indvars,depvars,
                                                               dict_indvars,dict_depvars, dict_depvar_input,
                                                               phi, derivative,chain,initθ,strategy,
                                                               eq_params=eq_params,
                                                               param_estim=param_estim,
                                                               default_p=default_p;
                                                               bc_indvars=bc_indvar,
                                                               integration_indvars=integration_indvar)
                                                               for (bc, bc_indvar, integration_indvar) in zip(bcs, bc_indvars, bc_integration_vars)]
    symbolic_pde_loss_functions, symbolic_bc_loss_functions
end

# Convert a PDE problem into an OptimizationProblem
function SciMLBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
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

    # TODO fix it in MTK 6.0.0+v: ModelingToolkit.get_ivs(pde_system)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(pde_system.ivs,
                                                         pde_system.dvs)

    chain = discretization.chain
    initθ = discretization.init_params
    flat_initθ = if (typeof(chain) <: AbstractVector) reduce(vcat, initθ) else  initθ end
    eltypeθ = eltype(flat_initθ)
    parameterless_type_θ =  DiffEqBase.parameterless_type(flat_initθ)

    flat_initθ = if param_estim == false flat_initθ else vcat(flat_initθ, adapt(typeof(flat_initθ), default_p)) end
        phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy
    if !(eqs isa Array)
    eqs = [eqs]
    end
    pde_indvars = if strategy isa QuadratureTraining
        get_argument(eqs, dict_indvars, dict_depvars)
   else
        get_variables(eqs, dict_indvars, dict_depvars)
    end
   pde_integration_vars = get_integration_variables(eqs, dict_indvars, dict_depvars)
    _pde_loss_functions = [build_loss_function(eq,indvars,depvars, 
                                               dict_indvars,dict_depvars,dict_depvar_input,
                                               phi, derivative,chain, initθ,strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                               bc_indvars=pde_indvar, integration_indvars=integration_indvar
                                               ) for (eq, pde_indvar, integration_indvar) in zip(eqs, pde_indvars, pde_integration_vars)]
    bc_indvars = if strategy isa QuadratureTraining
         get_argument(bcs, dict_indvars, dict_depvars)
    else
         get_variables(bcs, dict_indvars, dict_depvars)
    end
    bc_integration_vars = get_integration_variables(bcs, dict_indvars, dict_depvars)

    _bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                              dict_indvars,dict_depvars, dict_depvar_input,
                                              phi,derivative,chain,initθ,strategy;
                                              eq_params=eq_params,
                                              param_estim=param_estim,
                                              default_p=default_p,
                                              bc_indvars=bc_indvar,
                                              integration_indvars=integration_indvar) for (bc, bc_indvar, integration_indvar) in zip(bcs, bc_indvars, bc_integration_vars)]

    dict_span = Dict([Symbol(d.variables) => [d.domain.lower, d.domain.upper] for d in domains])                                         

    pde_bounds = map(pde_integration_vars) do eq_pde_integration_vars
        span = map(indvar -> dict_span[indvar], eq_pde_integration_vars)
    end
    bcs_bounds = map(bc_integration_vars) do eq_bc_integration_vars
        span = map(indvar -> dict_span[indvar], eq_bc_integration_vars)
    end

        pde_loss_functions, bc_loss_functions =
    if strategy isa GridTraining
        dx = strategy.dx

        train_sets = generate_training_sets(domains,dx,eqs,bcs,eltypeθ,
                                            dict_indvars,dict_depvars)

        # the points in the domain and on the boundary
        pde_train_sets, bcs_train_sets = train_sets

        pde_train_sets = adapt.(parameterless_type_θ, pde_train_sets)
        bcs_train_sets =  adapt.(parameterless_type_θ, bcs_train_sets)
        pde_loss_functions = [get_loss_function(_loss, _set, eltypeθ, parameterless_type_θ, strategy)
                                                for (_loss, _set) in zip(_pde_loss_functions, pde_train_sets)]

        bc_loss_functions =  [get_loss_function(_loss, _set, eltypeθ, parameterless_type_θ, strategy)
                                                for (_loss, _set) in zip(_bc_loss_functions, bcs_train_sets)]
        (pde_loss_functions, bc_loss_functions)
    elseif strategy isa StochasticTraining
          bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
          pde_bounds, bcs_bounds = bounds

          pde_loss_functions = [get_loss_function(_loss, bound, eltypeθ, parameterless_type_θ, strategy)
                                                  for (_loss, bound) in zip(_pde_loss_functions, pde_bounds)]

          strategy_ = StochasticTraining(strategy.bcs_points)
          bc_loss_functions = [get_loss_function(_loss, bound, eltypeθ, parameterless_type_θ, strategy_)
                                                 for (_loss, bound) in zip(_bc_loss_functions, bcs_bounds)]
          (pde_loss_functions, bc_loss_functions)
    elseif strategy isa QuasiRandomTraining
         bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
         pde_bounds, bcs_bounds = bounds

         pde_loss_functions = [get_loss_function(_loss, bound, eltypeθ, parameterless_type_θ, strategy)
                                                 for (_loss, bound) in zip(_pde_loss_functions, pde_bounds)]

         strategy_ = QuasiRandomTraining(strategy.bcs_points;
                                         sampling_alg=strategy.sampling_alg,
                                         resampling=strategy.resampling,
                                         minibatch=strategy.minibatch)
         bc_loss_functions = [get_loss_function(_loss, bound, eltypeθ, parameterless_type_θ, strategy_)
                                                for (_loss, bound) in zip(_bc_loss_functions, bcs_bounds)]
         (pde_loss_functions, bc_loss_functions)
    elseif strategy isa QuadratureTraining
        bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
        pde_bounds, bcs_bounds = bounds

        lbs, ubs = pde_bounds
        pde_loss_functions = [get_loss_function(_loss, lb, ub, eltypeθ, parameterless_type_θ, strategy)
                                                for (_loss, lb, ub) in zip(_pde_loss_functions, lbs, ubs)]
        lbs, ubs = bcs_bounds
        bc_loss_functions = [get_loss_function(_loss, lb, ub, eltypeθ, parameterless_type_θ, strategy)
                                               for (_loss, lb, ub) in zip(_bc_loss_functions, lbs, ubs)]

        (pde_loss_functions, bc_loss_functions)
    end

    pde_loss_function = θ -> sum(map(l -> l(θ), pde_loss_functions))
    bcs_loss_function = θ -> sum(map(l -> l(θ), bc_loss_functions))
    loss_function = θ -> pde_loss_function(θ) + bcs_loss_function(θ)

    function loss_function_(θ, p)
            if additional_loss isa Nothing
            return loss_function(θ)
        else
            function _additional_loss(phi, θ)
                θ_ = θ[1:end - length(default_p)]
                p = θ[(end - length(default_p) + 1):end]
                return additional_loss(phi, θ_, p)
            end
            return loss_function(θ) + _additional_loss(phi, θ)
        end
    end

    f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
    GalacticOptim.OptimizationProblem(f, flat_initθ)
end




@parameters t, x
@variables u1(..), u2(..), u3(..)
Dt = Differential(t)
Dtt = Differential(t)^2
Dx = Differential(x)
Dxx = Differential(x)^2

eqs = [Dtt(u1(t, x)) ~ Dxx(u1(t, x)) + u3(t, x) * sin(pi * x),
       Dtt(u2(t, x)) ~ Dxx(u2(t, x)) + u3(t, x) * cos(pi * x),
       0. ~ u1(t, x) * sin(pi * x) + u2(t, x) * cos(pi * x) - exp(-t)]

bcs = [u1(0, x) ~ sin(pi * x),
       u2(0, x) ~ cos(pi * x),
       Dt(u1(0, x)) ~ -sin(pi * x),
       Dt(u2(0, x)) ~ -cos(pi * x),
       u1(t, 0) ~ 0.,
       u2(t, 0) ~ exp(-t),
       u1(t, 1) ~ 0.,
       u2(t, 1) ~ -exp(-t)]


# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# Neural network
input_ = length(domains)
n = 15
chain = [FastChain(FastDense(input_, n, Flux.σ), FastDense(n, n, Flux.σ), FastDense(n, 1)) for _ in 1:3]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

_strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, _strategy, init_params=initθ)

pde_system = PDESystem(eqs, bcs, domains, [t,x], [u1,u2,u3])
prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
bcs_inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents

cb = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=5)


import ModelingToolkit: Interval, infimum, supremum
# 2d wave equation, neumann boundary condition
@parameters x, t
@variables u(..) w(..)
Dx = Differential(x)
Dtt = Differential(t)^2
Dt = Differential(t)
# 2D PDE
eq  = 0 ~ Dx(u(x)) + w(x, t)

# Initial and boundary conditions
bcs = [u(0) ~ 0.,
       u(1) ~ 0.,
       w(0, t) ~ 0,
       w(1, t) ~ exp(-t)]

# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
           t ∈ Interval(0.0, 1.0)]

# Neural network
chain = [FastChain(FastDense(2, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1)), FastChain(FastDense(1, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))]

eltypeθ = eltype(initθ[1])
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
flat_initθ = reduce(vcat, initθ)

_indvars = [x,t]
_depvars = [u(x), w(x, t)]
dim = length(domains)
dx = 0.1
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                     reltol=1e-3,abstol=1e-3,
                                                     maxiters=50, batch=100)
depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ[1])
phi = NeuralPDE.get_phi.(chain, parameterless_type_θ)

# map(phi_ -> phi_(rand(2,10), flat_initθ),phi)
derivative = NeuralPDE.get_numeric_derivative()

#= 
u_ = (cord, θ, phi) -> sum(phi(cord, θ))

phi([1,2], initθ)

phi_ = (p) -> phi(p, initθ)[1]
dphi = Zygote.gradient(phi_, [1.,2.])

dphi1 = derivative(phi, u_, [1.,2.], [[ 0.0049215667, 0.0]], 1, initθ)
dphi2 = derivative(phi, u_, [1.,2.], [[0.0,  0.0049215667]], 1, initθ)
isapprox(dphi[1][1], dphi1, atol=1e-8)
isapprox(dphi[1][2], dphi2, atol=1e-8) =#

_pde_loss_function = NeuralPDE.build_loss_function(eq, _indvars, _depvars, dict_depvar_input, phi, derivative, chain, initθ, strategy)

expr_pde_loss_function = NeuralPDE.build_symbolic_loss_function(eq, indvars, depvars, dict_depvar_input, phi, derivative, chain, initθ, strategy)

@show expr_pde_loss_function
#= expected
:((cord, var"##θ#529", phi, derivative, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  derivative.(phi, u, cord, Array{Float32,1}[[0.0, 0.0049215667], [0.0, 0.0049215667]], 2, var"##θ#529") .- derivative.(phi, u, cord, Array{Float32,1}[[0.0049215667, 0.0], [0.0049215667, 0.0]], 2, var"##θ#529")
              end
          end
      end) =#


bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars)
@show bc_indvars
#= expected
4-element Array{Array{Any,1},1}:
 [:t]
 [:t]
 [:x]
 [:x] =#


_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars, dict_depvar_input,
                                                     phi,derivative,chain,initθ,strategy,
                                                     bc_indvars=bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

expr_bc_loss_functions = [NeuralPDE.build_symbolic_loss_function(bc,indvars,depvars, dict_depvar_input,
                                                                        phi,derivative,chain,initθ,strategy,
                                                                        bc_indvars=bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]
#= 
4-element Array{Expr,1}:
 :((cord, var"##θ#529", phi, derivative, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  u.(cord, var"##θ#529", phi) .- 0.0
              end
          end
      end)
 :((cord, var"##θ#529", phi, derivative, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  u.(cord, var"##θ#529", phi) .- 0.0
              end
          end
      end)
 :((cord, var"##θ#529", phi, derivative, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  u.(cord, var"##θ#529", phi) .- (*).(x, (+).(1.0, (*).(-1, x)))
              end
          end
      end)
 :((cord, var"##θ#529", phi, derivative, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  derivative.(phi, u, cord, Array{Float32,1}[[0.0, 0.0049215667]], 1, var"##θ#529") .- 0.0
end
          end
      end) =#
train_sets = NeuralPDE.generate_training_sets(domains, dx, [eq], bcs, eltypeθ, indvars, depvars)
pde_train_set, bcs_train_set = train_sets

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains, [eq], bcs, eltypeθ, indvars, depvars, NeuralPDE.StochasticTraining(100))

@show pde_bounds
#= expected
1-element Vector{Vector{Any}}:
 [Float32[0.01, 0.99], Float32[0.01, 0.99]] =#

@show bcs_bounds
#= expected
4-element Vector{Vector{Any}}:
 [0, Float32[0.0, 1.0]]
 [1, Float32[0.0, 1.0]]
 [Float32[0.0, 1.0], 0]
 [Float32[0.0, 1.0], 0] =#

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)

pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
prob = NeuralPDE.discretize(pde_system, discretization)

expr_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
expr_pde_loss_function, expr_bc_loss_functions = expr_prob

cb = function (p, l)
println("loss: ", l)
    return false
end

res = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=5)



######### ZOE EXAMPLE
 @parameters x y
 @variables p(..) q(..) r(..) s(..)
 Dx = Differential(x)
 Dy = Differential(y)

 # 2D PDE
 eq  = p(x) + q(y) + Dx(r(x, y)) + s(x, y) ~ 0
 # eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

 # Initial and boundary conditions
 bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
         r(x, -1) ~ 0.f0, r(1, y) ~ 0.0f0, 
         s(x, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
 # bcs = [s(y, 1) ~ 0.0f0]
 # Space and time domains
 domains = [x ∈ IntervalDomain(0.0, 1.0),
y ∈ IntervalDomain(-1.0, 0.0)]

 # chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
 numhid = 3
 fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
               [FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
               reltol=1e-3,abstol=1e-3,
               maxiters=50, batch=100)
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                 strategy)

 pde_system = PDESystem(eq, bcs, domains, [x,y], [p(x), q(y), r(x, y), s(x,y)])


 sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
 prob = NeuralPDE.discretize(pde_system, discretization)
 initθ = discretization.init_params
 prob
 initθvec = vcat(initθ...)
 prob.f(initθvec, [])
 res = GalacticOptim.solve(prob, ADAM(0.1); maxiters=3)
 phi = discretization.phi
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
 depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(pde_system.indvars, pde_system.depvars)

chain = discretization.chain
 initθ = discretization.init_params
 flat_initθ = if (typeof(chain) <: AbstractVector) vcat(initθ...) else  initθ end
 flat_initθ = if param_estim == false flat_initθ else vcat(flat_initθ, adapt(DiffEqBase.parameterless_type(flat_initθ), default_p)) end
 phi = discretization.phi
 derivative = discretization.derivative
 strategy = discretization.strategy
 if !(eqs isa Array)
     eqs = [eqs]
    end
 pde_indvars = if strategy isa QuadratureTraining
NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
 else
         NeuralPDE.get_variables(eqs, dict_indvars, dict_depvars)
 end
 _pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,
                                             dict_indvars,dict_depvars,dict_depvar_input,
                                             phi, derivative,chain, initθ,strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                             bc_indvars=pde_indvar) for (eq, pde_indvar) in zip(eqs, pde_indvars)]
 bc_indvars = if strategy isa QuadratureTraining
         get_argument(bcs, dict_indvars, dict_depvars)
 else
         get_variables(bcs, dict_indvars, dict_depvars)
 end

 _bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                 dict_indvars,dict_depvars,dict_depvar_input,
                                                 phi, derivative,chain, initθ, strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p;
                                                 bc_indvars=bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)] 

