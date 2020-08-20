struct PhysicsInformedNN{D,C,P,PH,DER,A,T,K}
  dx::D
  chain::C
  initθ::P
  phi::PH
  autodiff::Bool
  derivative::DER
  additional_condtions::A
  training_strategies::T
  kwargs::K
end

function PhysicsInformedNN(dx,
                           chain,
                           init_params = nothing;
                           _phi = nothing,
                           autodiff=false,
                           additional_condtions = nothing,
                           _derivative = nothing,
                           training_strategies = TrainingStrategies(),
                           kwargs...)
    if init_params == nothing
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

    PhysicsInformedNN(dx,chain,initθ,phi,autodiff,derivative,additional_condtions,training_strategies, kwargs)
end

struct TrainingStrategies
    stochastic_loss::Bool
end

function TrainingStrategies(;stochastic_loss=true)
    TrainingStrategies(stochastic_loss)
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

# Calculate an order of derivative
function count_order(_args)
    n = 0
    while (_args[1] == :derivative)
        n += 1
        _args = _args[2].args
    end
    return n
end

function get_depvars(_args)
    while (_args[1] == :derivative)
        _args = _args[2].args
    end
    return _args[1]
end
function get_indpvars(_args)
    while (_args[1] == :derivative)
        _args = _args[2].args
    end
    return _args[2:end-1]
end

# Wrapper for _simplified_derivative
function simplified_derivative(ex,indvars,depvars,dict_indvars,dict_depvars)
    if ex isa Expr
        ex.args = _simplified_derivative(ex.args,indvars,depvars,dict_indvars,dict_depvars)
    end
    return ex
end

"""
Simplify the derivative expression

# Examples

1. First derivative of function 'u(x,y)' of variables: 'x, y'
1. First derivative of function 'u(x,y)' with respect to x

Take expressions in the form: `derivative(u(x,y,θ), x)` to `derivative(unn, x, y, xnn, order, θ)`,

where
 unn - unique number for the function
 x,y - coordinates of point
 xnn - unique number for the variable
 order - order of derivative
 θ - parameters of neural network
"""
function _simplified_derivative(_args,indvars,depvars,dict_indvars,dict_depvars)
    for (i,e) in enumerate(_args)
        if !(e isa Expr)
            if e == :derivative && _args[end] != :θ
                order = count_order(_args)
                depvars = get_depvars(_args)
                indvars = get_indpvars(_args)
                depvars_num = dict_depvars[depvars]
                _args = [:derivative,depvars_num, indvars...,dict_indvars[_args[3]], order, :θ]
                break
            end
        else
            _args[i].args = _simplified_derivative(_args[i].args,indvars,depvars,dict_indvars,dict_depvars)
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
     (derivative(1, x, y, 1, 2, θ) + derivative(1, x, y, 2, 2, θ)) - -(sin(πx)) * sin(πy)

3)  System of PDE: [Dx(u1(x,y,θ)) + 4*Dy(u2(x,y,θ)) ~ 0,
                    Dx(u2(x,y,θ)) + 9*Dy(u1(x,y,θ)) ~ 0]

    Take expressions in the form:
    2-element Array{Equation,1}:
        Equation(derivative(u1(x, y, θ), x) + 4 * derivative(u2(x, y, θ), y), ModelingToolkit.Constant(0))
        Equation(derivative(u2(x, y, θ), x) + 9 * derivative(u1(x, y, θ), y), ModelingToolkit.Constant(0))
    to
      [(derivative(1, x, y, 1, 1, θ) + 4 * derivative(2, x, y, 2, 1, θ)) - 0,
       (derivative(2, x, y, 1, 1, θ) + 9 * derivative(1, x, y, 2, 1, θ)) - 0]
"""
function parse_equation(eq,indvars,depvars,dict_indvars,dict_depvars)
    left_expr = simplified_derivative((ModelingToolkit.simplified_expr(eq.lhs)),
                                      indvars,depvars,dict_indvars,dict_depvars)
    right_expr = simplified_derivative((ModelingToolkit.simplified_expr(eq.rhs)),
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
                  [(derivative(1, x, y, 1, 1, θ) + 4 * derivative(2, x, y, 2, 1, θ)) - 0,
                   (derivative(2, x, y, 1, 1, θ) + 9 * derivative(1, x, y, 2, 1, θ)) - 0]
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

# get agruments from bondary condition functions
function get_bc_argument(bcs,indvars,depvars,dict_indvars,dict_depvars)
    bc_args = []
    for _bc in bcs
        if !(_bc isa Array)
            _bc = [_bc]
        end
        left_expr = simplified_derivative(ModelingToolkit.simplified_expr(_bc[1].lhs),
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
    if autodiff # automatic differentiation (not implemented yet)
        # derivative = (x,θ,n) -> ForwardDiff.gradient(x->phi(x,θ),x)[n]
    else # numerical differentiation
        epsilon = cbrt(eps(Float32))
        function get_ε(dim, der_num)
            ε = zeros(dim)
            ε[der_num] = epsilon
            ε
        end

        εs = [get_ε(dim,d) for d in 1:dim]

        derivative = (_x...) ->
        begin
            var_num = _x[1]
            x = collect(_x[2:end-3])
            der_num =_x[end-2]
            order = _x[end-1]
            θ = _x[end]
            ε = εs[der_num]
            return _derivative(x,θ,order,ε,der_num,var_num)
        end
        _derivative = (x,θ,order,ε,der_num,var_num) ->
        begin
            if order == 1
                #Five-point stencil
                # return (-phi(x+2ε,θ) + 8phi(x+ε,θ) - 8phi(x-ε,θ) + phi(x-2ε,θ))/(12*epsilon)
                if isuinplace
                    return (phi(x+ε,θ) - phi(x-ε,θ))/(2*epsilon)
                else
                    return (phi(x+ε,θ)[var_num] - phi(x-ε,θ)[var_num])/(2*epsilon)
                end
            else
                return (_derivative(x+ε,θ,order-1,ε,der_num,var_num)
                      - _derivative(x-ε,θ,order-1,ε,der_num,var_num))/(2*epsilon)
            end
        end
    end
    derivative
end

function get_loss_function(loss_function, train_set, τ, phi, derivative, training_strategies)

    stochastic_loss = training_strategies.stochastic_loss

    function inner_loss(loss_function,phi,x,θ,derivative)
        sum(loss_function(phi, x, θ, derivative))
    end

    if !(loss_function isa Array)
        loss_function = [loss_function]
        train_set = [train_set]
    end


    function loss(θ)
        if stochastic_loss
            include_frac = 0.75
            total = 0.
            for (j,l) in enumerate(loss_function)
                size_set = size(train_set[j])[1]
                count_elements = convert(Int64,round(include_frac*size_set, digits=0))
                if count_elements <= 2
                    count_elements = size_set
                end
                for i in 1:count_elements
                    index = convert(Int64, round(size_set*rand(1)[1] + 0.5, digits=0))
                    total += inner_loss(l,phi,train_set[j][index],θ,derivative)^2
                end
            end
            return (1.0f0/τ) * total
        else
            return (1.0f0/τ) *sum(sum(abs2,inner_loss(l,phi,x,θ,derivative)
                    for x in set) for (l,set) in zip(loss_function,train_set))
        end
    end
    return loss
end


# Convert a PDE problem into OptimizationProblem
function DiffEqBase.discretize(pde_system::PDESystem,discretization::PhysicsInformedNN)
    eqs = pde_system.eq
    bcs = pde_system.bcs
    if eqs isa Array
        for bc in bcs
            size(eqs) != size(bc) && error("PDE and Boundary conditions should have the same size")
        end
    end

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
    training_strategies = discretization.training_strategies

    train_sets = generate_training_sets(domains,dx,bcs,
                                        indvars,depvars,
                                        dict_indvars,dict_depvars)

    # the points in the domain and on the boundary
    train_domain_set,train_bound_set,train_set = train_sets

    expr_pde_loss_function = build_loss_function(eqs,indvars,depvars,
                                                 dict_indvars,dict_depvars)
    expr_bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                  dict_indvars,dict_depvars) for bc in bcs]

    # norm coefficient for loss function
    τ = sum(length(set) for set in train_domain_set)
    pde_loss_function = get_loss_function(eval(expr_pde_loss_function),
                                          train_domain_set,
                                          τ,
                                          phi,
                                          derivative,
                                          training_strategies)
    τ = sum(length(set) for set in train_bound_set)
    bc_loss_function = get_loss_function(eval.(expr_bc_loss_functions),
                                         train_bound_set,
                                         τ,
                                         phi,
                                         derivative,
                                         training_strategies)


    loss_functions = [pde_loss_function, bc_loss_function]
    additional_condtions = discretization.additional_condtions
    if !(additional_condtions == nothing)
        expr_add_loss_function = [build_loss_function(additional_condtion,
                                                     indvars,depvars,
                                                     dict_indvars,dict_depvars) for additional_condtion in additional_condtions]

        add_loss_function = get_loss_function(eval.(expr_add_loss_function),
                                              train_domain_set,
                                              1,
                                              phi,
                                              derivative,
                                              training_strategies)
        push!(loss_functions, add_loss_function)
    end

    function loss_function(θ,p)
        sum([lf(θ) for lf in loss_functions])
    end

    f = OptimizationFunction(loss_function, initθ, GalacticOptim.AutoZygote())
    GalacticOptim.OptimizationProblem(f, initθ)
end
