import Base.Broadcast

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

struct PhysicsInformedNN{isinplace,C,T,P,PH,DER,K} <: AbstractPINN{isinplace}
  chain::C
  strategy::T
  init_params::P
  phi::PH
  derivative::DER
  kwargs::K

 @add_kwonly function PhysicsInformedNN{iip}(chain,
                                             strategy;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
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

        if phi == nothing
            if chain isa AbstractArray
                _phi = get_phi.(chain)
            else
                _phi = get_phi(chain)
            end
        else
            _phi = phi
        end

        if derivative == nothing
            _derivative = get_numeric_derivative()
        else
            _derivative = derivative
        end
        new{iip,typeof(chain),typeof(strategy),typeof(initθ),typeof(_phi),typeof(_derivative),typeof(kwargs)}(chain,strategy,initθ,_phi,_derivative, kwargs)
    end
end
PhysicsInformedNN(chain,strategy,args...;kwargs...) = PhysicsInformedNN{true}(chain,strategy,args...;kwargs...)

SciMLBase.isinplace(prob::PhysicsInformedNN{iip}) where iip = iip


abstract type TrainingStrategies  end

"""
* `dx`: the discretization of the grid.
"""
struct GridTraining <: TrainingStrategies
    dx
end

"""
* `points`: number of points in random select training set.
"""
struct StochasticTraining <:TrainingStrategies
    points:: Int64
end

"""
* `points`:  the number of quasi-random points in minibatch,
* `sampling_alg`: the quasi-Monte Carlo sampling algorithm,
* `minibatch`: the number of subsets.

For more information look: QuasiMonteCarlo.jl https://github.com/SciML/QuasiMonteCarlo.jl
"""
struct QuasiRandomTraining <:TrainingStrategies
    points:: Int64
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
    minibatch:: Int64
end
function QuasiRandomTraining(points;sampling_alg = UniformSample(),minibatch=500)
    QuasiRandomTraining(points,sampling_alg,minibatch)
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

function QuadratureTraining(;quadrature_alg=HCubatureJL(),reltol= 1e-6,abstol= 1e-3,maxiters=1e3,batch=0)
    QuadratureTraining(quadrature_alg,reltol,abstol,maxiters,batch)
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
function transform_expression(ex,dict_indvars,dict_depvars)
    if ex isa Expr
        ex = _transform_expression(ex,dict_indvars,dict_depvars)
    end
    return ex
end

function get_ε(dim, der_num)
    epsilon = cbrt(eps(Float32))
    ε = zeros(Float32, dim)
    ε[der_num] = epsilon
    ε
end

θ = gensym("θ")

import ModelingToolkit:canonicalexpr,operation,arguments, toexpr,istree

function toexpr(O; canonicalize=true)
    if canonicalize
        canonical, O = canonicalexpr(O)
        canonical && return O
    else
        !istree(O) && return O
    end

    op = operation(O)
    args = arguments(O)
    if op isa Differential
        ex = toexpr(args[1]; canonicalize=canonicalize)
        wrt = toexpr(op.x; canonicalize=canonicalize)
        return :(_derivative($ex, $wrt))
    elseif op isa Sym
        isempty(args) && return nameof(op)
        return Expr(:call, toexpr(nameof(op); canonicalize=canonicalize), toexpr(args; canonicalize=canonicalize)...)
    end
    return Expr(:call, nameof(op), toexpr(args; canonicalize=canonicalize)...)
end


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
function _transform_expression(ex,dict_indvars,dict_depvars)
    _args = ex.args
    for (i,e) in enumerate(_args)
        if !(e isa Expr)
            if e in keys(dict_depvars)
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                ex.args = if length(dict_depvars) == 1
                    [:u, :set, :($θ), :phi]
                else
                    [:u, :set, Symbol(:($θ),num_depvar), Symbol(:phi,num_depvar)]
                end
                break
            elseif e == :_derivative
                derivative_variables = Symbol[]
                order = 0
                while (_args[1] == :_derivative)
                    order += 1
                    push!(derivative_variables, _args[end])
                    _args = _args[2].args
                end
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                dim_l = length(indvars)
                εs = [get_ε(dim_l,d) for d in 1:dim_l]
                undv = [dict_indvars[d_p] for d_p  in derivative_variables]
                εs_dnv = [εs[d] for d in undv]
                ex.args = if length(dict_depvars) == 1
                    [:derivative, :phi, :u, :set, εs_dnv, order, :($θ)]
                else
                    [:derivative, Symbol(:phi,num_depvar), :u, :set, εs_dnv, order, Symbol(:($θ),num_depvar)]
                end
                break
            end
        else
            ex.args[i] = _transform_expression(ex.args[i],dict_indvars,dict_depvars)
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

function build_symbolic_equation(eq,_indvars,_depvars)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    parse_equation(eq,dict_indvars,dict_depvars)
end


function parse_equation(eq,dict_indvars,dict_depvars)
    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)

    left_expr = Broadcast.__dot__(transform_expression(toexpr(eq_lhs ; canonicalize=false),
                                     dict_indvars,dict_depvars))
    right_expr = Broadcast.__dot__(transform_expression(toexpr(eq_rhs; canonicalize=false),
                                     dict_indvars,dict_depvars))

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
function build_symbolic_loss_function(eqs,_indvars,_depvars, phi, derivative,initθ; bc_indvars=nothing)
    # dictionaries: variable -> unique number
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars==nothing ? indvars : bc_indvars
    return build_symbolic_loss_function(eqs,indvars,depvars,
                                        dict_indvars,dict_depvars,
                                        phi, derivative,initθ,
                                        bc_indvars = bc_indvars)
end

function build_symbolic_loss_function(eqs,indvars,depvars,
                                      dict_indvars,dict_depvars,
                                      phi, derivative, initθ;
                                      bc_indvars = indvars)
    # if !(eqs isa Array)
    #     eqs = [eqs]
    # end
    # loss_functions= Expr[]
    # for eq in eqs
    #     push!(loss_functions,parse_equation(eq,dict_indvars,dict_depvars))
    # end
    loss_function = parse_equation(eqs,dict_indvars,dict_depvars)
    vars = :(cord, $θ, phi, derivative,u)
    ex = Expr(:block)
    if length(depvars) != 1
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

    push!(ex.args,  Expr(:(=),build_expr(:tuple, [:set, :cord_]), :cord))

    indvars_ex = [:($:cord_[$i]) for (i, u) ∈ enumerate(indvars)]


    left_arg_pairs, right_arg_pairs = indvars,indvars_ex
    vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
    let_ex = Expr(:let, vars_eq, loss_function)
    push!(ex.args,  let_ex)

    expr_loss_function = :(($vars) -> begin $ex end)
end

function build_loss_function(eqs,_indvars,_depvars, phi, derivative,initθ;bc_indvars=nothing)
    # dictionaries: variable -> unique number
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars==nothing ? indvars : bc_indvars
    return build_loss_function(eqs,indvars,depvars,
                               dict_indvars,dict_depvars,
                               phi, derivative,initθ,
                               bc_indvars = bc_indvars)
end

function build_loss_function(eqs,indvars,depvars,
                             dict_indvars,dict_depvars,
                             phi, derivative, initθ;
                             bc_indvars = indvars)

     expr_loss_function = build_symbolic_loss_function(eqs,indvars,depvars,
                                                       dict_indvars,dict_depvars,
                                                       phi, derivative, initθ;
                                                       bc_indvars = bc_indvars)
    u = get_u()
    derivative_ = get_numeric_derivative()
    _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
    loss_function = (cord, θ) -> _loss_function(cord, θ, phi, derivative_, u)
    return loss_function
end

function get_vars(indvars_, depvars_)
    depvars = [nameof(value(d)) for d in depvars_]
    indvars = [nameof(value(i)) for i in indvars_]
    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return depvars,indvars,dict_indvars,dict_depvars
end

function get_bc_varibles(bcs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return get_bc_varibles(bcs,dict_indvars,dict_depvars)
end

function get_bc_varibles(bcs,dict_indvars,dict_depvars)
    bc_args = get_bc_argument(bcs,dict_indvars,dict_depvars)
    return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
end

# function is_thing_in_expr(ex::Expr, thing)
#     local ans = false
#     for e in ex.args
#         if e isa Expr
#             ans = is_thing_in_expr(e,thing)
#         end
#         e == thing && return true
#     end
#     return ans
# end

function find_thing_in_expr(ex::Expr, thing; ans = Expr[])
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
function get_bc_argument(bcs,dict_indvars,dict_depvars)
    bcs_expr = toexpr.(bcs)
    vars = map(bcs_expr) do bc_expr
        _vars =  map(depvar -> find_thing_in_expr(bc_expr,  depvar), collect(keys(dict_depvars)))
        f_vars = filter(x -> !isempty(x), _vars)
        map(x -> first(x), f_vars)
    end
    bc_args = map(vars) do _vars
        map(var -> var.args[2:end] , _vars)
    end
    return first.(bc_args) #TODO for all arguments
end

function generate_training_sets(domains,dx,bcs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return generate_training_sets(domains,dx,bcs,dict_indvars,dict_depvars)
end
# Generate training set in the domain and on the boundary
function generate_training_sets(domains,dx,bcs,dict_indvars::Dict,dict_depvars::Dict)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx,length(domains))
    end

    bound_args = get_bc_argument(bcs,dict_indvars,dict_depvars)
    bound_vars = get_bc_varibles(bcs,dict_indvars,dict_depvars)

    spans = [d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)]
    dict_var_span = Dict([Symbol(d.variables) => d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)])

    dif = [Float64[] for i=1:size(domains)[1]]
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
    # train_set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(spans...)))...))
    #TODO adapt(typeof(θ) all set
    pde_train_set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(bc_data...)))...))
    pde_cord = [collect(pde_train_set[i,:]') for i  in 1:size(pde_train_set)[1]]

    # bcs_train_set = map(bound_vars) do bt #for Quadrature strategy
    #     span = map(b -> dict_var_span[b], bt)
    #     _set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    # end
    bcs_train_sets = map(bound_args) do bt #for dots strategy
        span = map(b -> get(dict_var_span, b, b), bt)
        _set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end
    bcs_cords = [[collect(bcs_train_set[i,:]') for i in 1:size(bcs_train_set)[1]] for bcs_train_set in bcs_train_sets]
    #TODO for Dt(u(t)) ~ Dx(v(x, t))

    # [pde_train_set,bcs_train_set,train_set]
      [[pde_train_set,pde_cord],[bcs_train_sets,bcs_cords]]
end

function get_bounds(domains,bcs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return get_bounds(domains,bcs,dict_indvars,dict_depvars)
end

function get_bounds(domains,bcs,dict_indvars,dict_depvars)
    bound_vars = get_bc_varibles(bcs,dict_indvars,dict_depvars)

    pde_lower_bounds = [d.domain.lower for d in domains]
    pde_upper_bounds = [d.domain.upper for d in domains]
    pde_bounds= [pde_lower_bounds,pde_upper_bounds]

    dict_lower_bound = Dict([Symbol(d.variables) => d.domain.lower for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => d.domain.upper for d in domains])

    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds= [bcs_lower_bounds,bcs_upper_bounds]

    [pde_bounds, bcs_bounds]
end


function get_phi(chain)
    # The phi trial solution
    if chain isa FastChain
        phi = (x,θ) -> chain(adapt(typeof(θ),x),θ)
    else
        _,re  = Flux.destructure(chain)
        phi = (x,θ) -> re(θ)(adapt(typeof(θ),x))
    end
    phi
end

function get_u()
	u = (cord, θ, phi)->phi(cord, θ)
end


Base.Broadcast.broadcasted(::typeof(get_u()), cord, θ, phi) = get_u()(cord, θ, phi)

# the method to calculate the derivative
function get_numeric_derivative()
    _epsilon = 1/(2*cbrt(eps(Float32)))
    derivative = (phi,u,x,εs,order,θ) ->
    begin
        ε = εs[order]
        if order > 1
            return (derivative(phi,u,x .+ ε,εs,order-1,θ)
                  - derivative(phi,u,x .- ε,εs,order-1,θ))*_epsilon
        else
            ε = adapt(typeof(θ),ε)
            x = adapt(typeof(θ),x)
            return (u(x .+ ε,θ,phi) - u(x .- ε,θ,phi))*_epsilon
        end
    end
    derivative
end

Base.Broadcast.broadcasted(::typeof(get_numeric_derivative()), phi,u,x,εs,order,θ) = get_numeric_derivative()(phi,u,x,εs,order,θ)

function get_loss_function(loss_functions, train_sets, strategy::GridTraining;τ=nothing)
    # norm coefficient for loss function
    if τ == nothing
        τ_ = loss_functions isa Array ? sum(length(train_set) for train_set in train_sets) : length(train_sets)
        τ = 1.0f0 / τ_
    end

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        train_sets = [train_sets]
    end
    f = (loss,train_set,θ) -> sum([inner_loss(loss,x,θ) for x in train_set])
    loss = (θ) ->  τ * sum(f(loss_function,train_set,θ) for (loss_function,train_set) in zip(loss_functions,train_sets))
    return loss
end

function get_loss_function_gpu(loss_functions, train_sets, strategy::GridTraining;τ=nothing)
    if τ == nothing
        τ_ = sum(size(set[1])[2] for set in train_sets)
        τ = 1.0f0 / τ_
    end
    loss = (θ) -> τ * sum(sum(abs2,loss_function(train_set, θ)) for (loss_function,train_set) in zip(loss_functions,train_sets))
    return loss
end



function get_loss_function(loss_functions, bounds, strategy::StochasticTraining;τ=nothing)
    points = strategy.points
    lbs,ubs = bounds

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        lbs = [lbs]
        ubs = [ubs]
    end
    if τ == nothing
        τ = 1.0f0 / points
    end

    loss = (θ) -> begin
        total = 0.
        for (lb, ub,l) in zip(lbs, ubs, loss_functions)
            len = length(lb)
            for i in 1:points
                r_point = lb .+ ub .* rand(len)
                total += inner_loss(l,r_point,θ)
            end
        end
        return τ * total
    end
    return loss
end

function get_loss_function(loss_functions, bounds, strategy::QuasiRandomTraining;τ=nothing)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    minibatch = strategy.minibatch
    lbs,ubs = bounds

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        lbs = [lbs]
        ubs = [ubs]
    end
    if τ == nothing
        τ = 1.0f0 / points
    end
    ss =[]
    for (lb, ub) in zip(lbs, ubs)
        s = QuasiMonteCarlo.generate_design_matrices(points,lb,ub,sampling_alg,minibatch)
        push!(ss,s)
    end
    loss = (θ) -> begin
        total = 0.
        for (lb, ub,s_,l) in zip(lbs,ubs,ss,loss_functions)
            s =  s_[rand(1:minibatch)]
            step_ = size(lb)[1]
            k = size(lb)[1]-1
            for i in 1:step_:step_*points
                r_point = s[i:i+k]
                total += inner_loss(l,r_point,θ)
            end
        end
        return τ * total
    end
    return loss
end

function get_loss_function(loss_functions, bounds, strategy::QuadratureTraining;τ=nothing)
    lbs,ubs = bounds
    if τ == nothing
        τ = 1.0f0
    end

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        lbs = [lbs]
        ubs = [ubs]
    end

    f = (lb,ub,loss_,θ) -> begin
        _loss = (x,θ) -> inner_loss(loss_, x, θ)
        prob = QuadratureProblem(_loss,lb,ub,θ;batch = strategy.batch)
        abs(solve(prob,
              strategy.quadrature_alg,
              reltol = strategy.reltol,
              abstol = strategy.abstol,
              maxiters = strategy.maxiters)[1])
    end
    loss = (θ) -> τ*sum(f(lb,ub,loss_,θ) for (lb,ub,loss_) in zip(lbs,ubs,loss_functions))
    return loss
end
function symbolic_discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs

    domains = pde_system.domain
    # dimensionality of equation
    dim = length(domains)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(pde_system.indvars, pde_system.depvars)

    chain = discretization.chain
    initθ = discretization.init_params
    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy

    symbolic_pde_loss_functions = [build_symbolic_loss_function(eq,indvars,depvars,
                                                              dict_indvars,dict_depvars,
                                                              phi, derivative,initθ) for eq  in eqs]

    bc_indvars = get_bc_varibles(bcs,dict_indvars,dict_depvars)
    symbolic_bc_loss_functions = [build_symbolic_loss_function(bc,indvars,depvars,
                                                               dict_indvars,dict_depvars,
                                                               phi, derivative,initθ;
                                                               bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
    symbolic_pde_loss_functions,symbolic_bc_loss_functions
end

# Convert a PDE problem into an OptimizationProblem
function DiffEqBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs

    domains = pde_system.domain
    # dimensionality of equation
    dim = length(domains)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(pde_system.indvars,pde_system.depvars)

    chain = discretization.chain
    initθ = discretization.init_params
    flat_initθ = if length(depvars) != 1 vcat(initθ...) else  initθ end
    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy

    _pde_loss_functions = [build_loss_function(eq,indvars,depvars,
                                             dict_indvars,dict_depvars,
                                             phi, derivative, initθ) for eq in eqs]

    bc_indvars = get_bc_varibles(bcs,dict_indvars,dict_depvars)
    _bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                  dict_indvars,dict_depvars,
                                                  phi, derivative, initθ;
                                                  bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    pde_loss_function, bc_loss_function =
    if strategy isa GridTraining
        dx = strategy.dx

        train_sets = generate_training_sets(domains,dx,bcs,
                                            dict_indvars,dict_depvars)

        # the points in the domain and on the boundary
        pde_train_set, bcs_train_sets = train_sets
        pde_set, pde_cord = pde_train_set
        bcs_set, bcs_cords =bcs_train_sets
        bcs_train_sets_ = [[bcs_set[i], bcs_cords[i]] for i in 1:length(bcs_cords)]

        pde_loss_function = get_loss_function_gpu(_pde_loss_functions,
                                                        [pde_train_set],
                                                        strategy)

        bc_loss_function = get_loss_function_gpu(_bc_loss_functions,
                                                       bcs_train_sets_,
                                                       strategy)
        (pde_loss_function, bc_loss_function)
    elseif strategy isa StochasticTraining
          bounds = get_bounds(domains,bcs,dict_indvars,dict_depvars)
          pde_bounds, bcs_bounds = bounds
          pde_loss_function = get_loss_function(_pde_loss_function,
                                                          pde_bounds,
                                                          strategy)
          plbs,pubs = pde_bounds
          blbs,bubs = bcs_bounds
          pl = length(plbs)
          bl = length(blbs[1])
          bsl = length(blbs)
          points = length(blbs[1]) == 0 ? 1 : bsl*Int(round(strategy.points^(bl/pl)))
          strategy = StochasticTraining(points)

          bc_loss_function = get_loss_function(_bc_loss_functions,
                                                         bcs_bounds,
                                                         strategy)
          (pde_loss_function, bc_loss_function)
    elseif strategy isa QuasiRandomTraining
         bounds = get_bounds(domains,bcs,dict_indvars,dict_depvars)
         pde_bounds, bcs_bounds = bounds
         pde_loss_function = get_loss_function(_pde_loss_function,
                                                        pde_bounds,
                                                        strategy)

         plbs,pubs = pde_bounds
         blbs,bubs = bcs_bounds
         pl = length(plbs)
         bl = length(blbs[1])
         bsl = length(blbs)
         points = length(blbs[1]) == 0 ? 1 : bsl*Int(round(strategy.points^(bl/pl)))
         strategy = QuasiRandomTraining(points;
                                        sampling_alg = strategy.sampling_alg,
                                        minibatch = strategy.minibatch)

         bc_loss_function = get_loss_function(_bc_loss_functions,
                                                       bcs_bounds,
                                                       strategy)
         (pde_loss_function, bc_loss_function)
    elseif strategy isa QuadratureTraining
        bounds = get_bounds(domains,bcs,dict_indvars,dict_depvars)
        pde_bounds, bcs_bounds = bounds

        plbs,pubs = pde_bounds
        blbs,bubs = bcs_bounds
        pl = length(plbs)
        bl = length(blbs[1])
        bsl = length(blbs)

        τ_ = (10)^length(plbs)
        τp = 1.0f0 / τ_

        pde_loss_function = get_loss_function(_pde_loss_function,
                                              pde_bounds,
                                              strategy;
                                              τ=τp)

        τb =  1.0f0 / (bsl * τ_^(bl/pl))

        if bl == 0
            bc_loss_function = get_loss_function(_bc_loss_functions,
                                                 [[[]]],
                                                 GridTraining(0.1))

        elseif bl == 1 && strategy.quadrature_alg in [CubaCuhre(),CubaDivonne()]
            @warn "$(strategy.quadrature_alg) does not work with one-dimensional
            problems, so for the boundary conditions loss function,
            the quadrature algorithm was replaced by HCubatureJL"

            strategy = QuadratureTraining(quadrature_alg = HCubatureJL(),
                                          reltol = bsl*(strategy.reltol)^(bl/pl),
                                          abstol = bsl*(strategy.abstol)^(bl/pl),
                                          maxiters = bsl*Int(round((strategy.maxiters)^(bl/pl))),
                                          batch = strategy.batch)
            bc_loss_function = get_loss_function(_bc_loss_functions,
                                                 bcs_bounds,
                                                 strategy;
                                                 τ=τb)
        else
            strategy = QuadratureTraining(quadrature_alg = strategy.quadrature_alg,
                                          reltol = bsl*(strategy.reltol)^(bl/pl),
                                          abstol = bsl*(strategy.abstol)^(bl/pl),
                                          maxiters = bsl*Int(round((strategy.maxiters)^(bl/pl))),
                                          batch = strategy.batch)
            bc_loss_function = get_loss_function(_bc_loss_functions,
                                             bcs_bounds,
                                             strategy;
                                             τ=τb)
        end

        (pde_loss_function, bc_loss_function)
    end

    function loss_function_(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
    GalacticOptim.OptimizationProblem(f, flat_initθ)
end
