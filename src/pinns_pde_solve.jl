"""
Algorithm for solving Physics Informed Neural Networks problem.

Arguments:
* `dx` is discretization of grid
* `chain` is a Flux.jl chain with d dimensional input and 1 dimensional output,
* `init_params` is the initial parameter of the neural network,
* `phi` is trial solution,
* `autodiff` is a boolean variable that determines whether to use automatic, differentiation(not supported while) or numerical,
* `derivative` is method that calculate derivative,
* `strategy` is determines which training strategy will be used.
"""

struct PhysicsInformedNN{D,C,P,PH,DER,T,K}
  dx::D
  chain::C
  initθ::P
  phi::PH
  autodiff::Bool
  derivative::DER
  strategy::T
  kwargs::K
end

function PhysicsInformedNN(dx,
                           chain,
                           init_params = nothing;
                           _phi = nothing,
                           autodiff=false,
                           _derivative = nothing,
                           strategy = GridTraining(),
                           kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    isuinplace = chain.layers[end].out == 1
    dim = chain.layers[1].in

    if _phi == nothing
        phi = get_phi(chain,isuinplace)
    else
        phi = _phi
    end

    if _derivative == nothing
        derivative = get_derivative(dim,phi,autodiff,isuinplace)
    else
        derivative = _derivative
    end

    PhysicsInformedNN(dx,chain,initθ,phi,autodiff,derivative, strategy, kwargs)
end

abstract type TrainingStrategies  end

struct GridTraining <: TrainingStrategies end

struct StochasticTraining <:TrainingStrategies
    include_frac::Float64
end
function StochasticTraining(;include_frac=0.75)
    StochasticTraining(include_frac)
end

struct QuadratureTraining <: TrainingStrategies
    algorithm::DiffEqBase.AbstractQuadratureAlgorithm
    reltol::Float64
    abstol::Float64
    maxiters::Int64
end
function QuadratureTraining(;algorithm=HCubatureJL(),reltol=1e-2,abstol=1e-2,maxiters=3)
    QuadratureTraining(algorithm,reltol,abstol,maxiters)
end

struct AdaptiveMonteCarloTraning <: TrainingStrategies end

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

# Wrapper for _transform_derivative
function transform_derivative(ex,indvars,depvars,dict_indvars,dict_depvars)
    if ex isa Expr
        ex.args = _transform_derivative(ex.args,indvars,depvars,dict_indvars,dict_depvars)
    end
    return ex
end

"""
Transform the derivative expression to inner representation

# Examples

1. First derivative of function 'u(x,y)' with respect to x

Take expressions in the form: `derivative(u(x,y,θ), x)` to `derivative(unn, x, y, undv, order, θ)`,

where
 unn - unique number for the function
 x,y - coordinates of point
 undv - uniqie number for derivative variables
 order - order of derivative
 θ - parameters of neural network
"""
function _transform_derivative(_args,indvars,depvars,dict_indvars,dict_depvars)
    for (i,e) in enumerate(_args)
        if !(e isa Expr)
            if e == :derivative && _args[end] != :θ
                derivative_variables = []
                order = 0
                while (_args[1] == :derivative)
                    order += 1
                    push!(derivative_variables, _args[end])
                    _args = _args[2].args
                end
                depvar = _args[1]
                indvars = _args[2:end-1]
                depvars_num = dict_depvars[depvar]
                undv = [dict_indvars[d_p] for d_p  in derivative_variables]
                _args = [:derivative,depvars_num, indvars...,undv, order, :θ]
                break
            end
        else
            _args[i].args = _transform_derivative(_args[i].args,indvars,depvars,dict_indvars,dict_depvars)
        end
    end
    return _args
end

"""
Parse ModelingToolkit equation form to the inner representation.

Example:

1)  1d ODE: Dt(u(t,θ)) ~ t +1

    Take expressions in the form: 'Equation(derivative(u(t, θ), t), t + 1)' to 'derivative(u, t, 1, 1, θ) - (t + 1)'

2)  2d PDE: Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

    Take expressions in the form:
     Equation(derivative(derivative(u(x, y, θ), x), x) + derivative(derivative(u(x, y, θ), y), y), -(sin(πx)) * sin(πy))
    to
     (derivative(1, x, y, [[ε,0],[ε,0]], 2, θ) + derivative(1, x, y, [[0,ε],[0,ε]], 2, θ)) - -(sin(πx)) * sin(πy)

3)  System of PDE: [Dx(u1(x,y,θ)) + 4*Dy(u2(x,y,θ)) ~ 0,
                    Dx(u2(x,y,θ)) + 9*Dy(u1(x,y,θ)) ~ 0]

    Take expressions in the form:
    2-element Array{Equation,1}:
        Equation(derivative(u1(x, y, θ), x) + 4 * derivative(u2(x, y, θ), y), ModelingToolkit.Constant(0))
        Equation(derivative(u2(x, y, θ), x) + 9 * derivative(u1(x, y, θ), y), ModelingToolkit.Constant(0))
    to
      [(derivative(1, x, y, [[ε,0]], 1, θ) + 4 * derivative(2, x, y, [[0,ε]], 1, θ)) - 0,
       (derivative(2, x, y, [[ε,0]], 1, θ) + 9 * derivative(1, x, y, [[0,ε]], 1, θ)) - 0]
"""
function parse_equation(eq,indvars,depvars,dict_indvars,dict_depvars)
    left_expr = transform_derivative(ModelingToolkit.simplified_expr(eq.lhs),
                                      indvars,depvars,dict_indvars,dict_depvars)
    right_expr = transform_derivative(ModelingToolkit.simplified_expr(eq.rhs),
                                       indvars,depvars,dict_indvars,dict_depvars)
    loss_func = :($left_expr - $right_expr)
end

"""
Build loss function for pde or boundary condition

# Examples: System of pde:

Take expressions in the form:

[Dx(u1(x,y,θ)) + 4*Dy(u2(x,y,θ)) ~ 0,
 Dx(u2(x,y,θ)) + 9*Dy(u1(x,y,θ)) ~ 0]

to

:((phi, vec, θ, derivative)->begin
          #= none:33 =#
          #= none:33 =#
          begin
              begin
                  u1 = ((x, y, θ)->begin
                              #= none:18 =#
                              (phi([x, y], θ))[1]
                          end)
                  u2 = ((x, y, θ)->begin
                              #= none:18 =#
                              (phi([x, y], θ))[2]
                          end)
              end
              let (x, y) = (vec[1], vec[2])
                  [(derivative(1, x, y, [[ε,0]], 1, θ) + 4 * derivative(2, x, y, [[0,ε]], 1, θ)) - 0,
                   (derivative(2, x, y, [[ε,0]], 1, θ) + 9 * derivative(1, x, y, [[0,ε]], 1, θ)) - 0]
              end
          end
      end)
"""
function build_loss_function(eqs,_indvars,_depvars)
    # dictionaries: variable -> unique number
    indvars = Expr.(_indvars)
    depvars = [d.name for d in _depvars]
    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return build_loss_function(eqs,indvars,depvars,dict_indvars,dict_depvars)
end

function build_loss_function(eqs,indvars,depvars,dict_indvars,dict_depvars)
    if !(eqs isa Array)
        eqs = [eqs]
    end
    loss_functions= []
    for eq in eqs
        loss_function = parse_equation(eq,depvars,indvars,dict_indvars,dict_depvars)
        push!(loss_functions,loss_function)
    end

    vars = :(phi, cord, θ, derivative)
    ex = Expr(:block)
    us = []
    for v in depvars
        var_num = dict_depvars[v]
        push!(us,:($v = ($(indvars...), θ) -> phi([$(indvars...)],θ)[$var_num]))
    end

    u_ex = ModelingToolkit.build_expr(:block, us)
    push!(ex.args,  u_ex)

    indvars_ex = [:($:cord[$i]) for (i, u) ∈ enumerate(indvars)]
    arg_pairs_indvars = indvars,indvars_ex

    left_arg_pairs, right_arg_pairs = arg_pairs_indvars
    vars_eq = Expr(:(=), ModelingToolkit.build_expr(:tuple, left_arg_pairs),
                            ModelingToolkit.build_expr(:tuple, right_arg_pairs))
    let_ex = Expr(:let, vars_eq, ModelingToolkit.build_expr(:vect, loss_functions))
    push!(ex.args,  let_ex)

    return :(($vars) -> begin $ex end)
end

# Get agruments from bondary condition functions
function get_bc_argument(bcs,indvars,depvars,dict_indvars,dict_depvars)
    bc_args = []
    for _bc in bcs
        _bc isa Array && error("boundary conditions must be represented as a one-dimensional array")
        left_expr = transform_derivative(ModelingToolkit.simplified_expr(_bc.lhs),
                                          indvars,depvars,dict_indvars,dict_depvars)
        bc_arg = if (left_expr.args[1] == :derivative)
            left_expr.args[3:end-3]
        else
            left_expr.args[2:end-1]
        end
        push!(bc_args, bc_arg)
    end
    return bc_args
end

function generate_training_sets(domains,dx,bcs,_indvars,_depvars)
    indvars = Expr.(_indvars)
    depvars = [d.name for d in _depvars]
    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return generate_training_sets(domains,dx,bcs,indvars,depvars,dict_indvars,dict_depvars)
end
# Generate training set in the domain and on the boundary
function generate_training_sets(domains,dx,bcs,indvars,depvars,dict_indvars,dict_depvars)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx,length(domains))
    end

    bound_args = get_bc_argument(bcs,indvars,depvars,dict_indvars,dict_depvars)

    spans = [d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)]
    dict_var_span = Dict([Symbol(d.variables) => d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)])

    train_set = []
    for points in Iterators.product(spans...)
        push!(train_set, [points...])
    end

    train_bound_set = []
    for bt in bound_args
        _set = []
        span = [get(dict_var_span, b, b) for b in bt]
        for points in Iterators.product(span...)
            push!(_set, [points...])
        end
        push!(train_bound_set, _set)
    end

    flat_train_bound_set = collect(Iterators.flatten(train_bound_set))
    train_domain_set =  setdiff(train_set, flat_train_bound_set)

    [train_domain_set,train_bound_set,train_set]
end

function get_phi(chain,isuinplace)
    # The phi trial solution
    if chain isa FastChain
        if isuinplace
            phi = (x,θ) -> first(chain(adapt(typeof(θ),x),θ))
        else
            phi = (x,θ) -> chain(adapt(typeof(θ),x),θ)
        end
    else
        _,re  = Flux.destructure(chain)
        if isuinplace
            phi = (x,θ) -> first(re(θ)(adapt(typeof(θ),x)))
        else
            phi = (x,θ) -> re(θ)(adapt(typeof(θ),x))
        end
    end
    phi
end

function get_derivative(dim,phi,autodiff,isuinplace)
    epsilon = cbrt(eps(Float32))
     function get_ε(dim, der_num)
         ε = zeros(dim)
         ε[der_num] = epsilon
         ε
     end
     εs = [get_ε(dim,d) for d in 1:dim]

    if autodiff # automatic differentiation (not implemented yet)
        error("automatic differentiation is not implemented yet)")
    else # numerical differentiation
        derivative = (_x...) ->
        begin
            var_num = _x[1]
            x = collect(_x[2:end-3])
            dnv =_x[end-2]
            order = _x[end-1]
            θ = _x[end]
            εs_dnv = [εs[d] for d in dnv]
            return _derivative(x,θ,order,εs_dnv,var_num)
        end
        _derivative = (x,θ,order,εs_dnv,var_num) ->
        begin
            ε = εs_dnv[order]
            if order == 1
                if isuinplace
                    return (phi(x+ε,θ) - phi(x-ε,θ))/(2*epsilon)
                else
                    return (phi(x+ε,θ)[var_num] - phi(x-ε,θ)[var_num])/(2*epsilon)
                end
            else
                return (_derivative(x+ε,θ,order-1, εs_dnv, var_num)
                      - _derivative(x-ε,θ,order-1, εs_dnv, var_num))/(2*epsilon)
            end
        end
    end
    derivative
end

function get_loss_function(loss_function, train_set, domains, phi, derivative, strategy)
    # norm coefficient for loss function
    τ = sum(length(set) for set in train_set)

    function inner_loss(loss_function,phi,x,θ,derivative)
        sum(loss_function(phi, x, θ, derivative))
    end

    if !(loss_function isa Array)
        loss_function = [loss_function]
        train_set = [train_set]
    end

    if strategy isa StochasticTraining
        include_frac = strategy.include_frac
        count_elements = []
        sets_size = []
        for j in 1:length(train_set)
            size_set = size(train_set[j])[1]
            count_element = convert(Int64,round(include_frac*size_set, digits=0))
            if count_element <= 2
                count_element = size_set
            end
            push!(sets_size,size_set)
            push!(count_elements,count_element)
        end
        loss = (θ) -> begin
            total = 0.
            for (j,l) in enumerate(loss_function)
                size_set = sets_size[j]
                for i in 1:count_elements[j]
                    index = convert(Int64, round(size_set*rand(1)[1] + 0.5, digits=0))
                    total += inner_loss(l,phi,train_set[j][index],θ,derivative)^2
                end
            end
            return (1.0f0/τ) * total
        end

    elseif strategy isa GridTraining
        loss = (θ) -> begin
            return (1.0f0/τ) *sum(sum(abs2,inner_loss(l,phi,x,θ,derivative)
                    for x in set) for (l,set) in zip(loss_function,train_set))
        end
    elseif strategy isa QuadratureTraining
        lb = [domain.domain.lower for domain in domains]
        ub = [domain.domain.upper  for domain in domains]
        f = (lb,ub,loss_,θ) -> begin
            _loss = (x,θ) -> sum(abs2,inner_loss(loss_, phi, x, θ, derivative))
            prob = QuadratureProblem(_loss,lb,ub,θ)
            solve(prob,
                  strategy.algorithm,
                  reltol = strategy.reltol,
                  abstol = strategy.abstol,
                  maxiters = strategy.maxiters)[1]
        end
        loss = (θ) -> (1.0f0/τ)*sum(f(lb,ub,loss_,θ) for loss_ in loss_function)
    end
    return loss
end

# Convert a PDE problem into OptimizationProblem
function DiffEqBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eq
    bcs = pde_system.bcs

    domains = pde_system.domain
    # dimensionality of equation
    dim = length(domains)
    dim > 3 && error("While only dimensionality no more than 3")

    depvars = [d.name for d in pde_system.depvars]
    indvars = Expr.(pde_system.indvars)
    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)

    dx = discretization.dx
    chain = discretization.chain
    initθ = discretization.initθ
    phi = discretization.phi
    autodiff = discretization.autodiff
    derivative = discretization.derivative
    autodiff == true && error("Automatic differentiation is not support yet")
    strategy = discretization.strategy

    train_sets = generate_training_sets(domains,dx,bcs,
                                        indvars,depvars,
                                        dict_indvars,dict_depvars)

    # the points in the domain and on the boundary
    train_domain_set,train_bound_set,train_set = train_sets

    expr_pde_loss_function = build_loss_function(eqs,indvars,depvars,
                                                 dict_indvars,dict_depvars)
    expr_bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                  dict_indvars,dict_depvars) for bc in bcs]

    pde_loss_function = get_loss_function(eval(expr_pde_loss_function),
                                          train_domain_set,
                                          domains,
                                          phi,
                                          derivative,
                                          strategy)
    bc_loss_function = get_loss_function(eval.(expr_bc_loss_functions),
                                         train_bound_set,
                                         domains,
                                         phi,
                                         derivative,
                                         strategy)

    function loss_function(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    f = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
    GalacticOptim.OptimizationProblem(f, initθ)
end
