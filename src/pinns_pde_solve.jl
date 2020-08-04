struct PhysicsInformedNN{int}
  dx::int
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
            # if autodiff
            #     error("While autodiff not support")
            # else
            # end
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
build loss function for pde or boundary function

# Examples

Take expressions in the form:
2-element Array{Equation,1}:
 Equation(derivative(u1(x, y, θ), x) + 4 * derivative(u2(x, y, θ), y), ModelingToolkit.Constant(0))
 Equation(derivative(u2(x, y, θ), x) + 9 * derivative(u1(x, y, θ), y), ModelingToolkit.Constant(0))

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
function build_loss_function(_funcs,vars,depvars,indvars,dict_depvars)
    ex = Expr(:block)

    us = []
    for v in depvars
        var_num = dict_depvars[v]
        push!(us,:($v = ($(indvars...), θ) -> phi([$(indvars...)],θ)[$var_num]))
    end

    u_ex = ModelingToolkit.build_expr(:block, us)
    push!(ex.args,  u_ex)

    indvars_ex = [:($:vec[$i]) for (i, u) ∈ enumerate(indvars)]
    arg_pairs_indvars = indvars,indvars_ex

    left_arg_pairs, right_arg_pairs = arg_pairs_indvars
    vars_eq = Expr(:(=), ModelingToolkit.build_expr(:tuple, left_arg_pairs),
                            ModelingToolkit.build_expr(:tuple, right_arg_pairs))
    let_ex = Expr(:let, vars_eq, ModelingToolkit.build_expr(:vect, _funcs))
    push!(ex.args,  let_ex)

    return :(($vars) -> begin $ex end)
end

"""
Extract ModelingToolkit PDE form to the inner representation.

Example:

1)  Equation: Dt(u(t,θ)) ~ t +1

    Take expressions in the form: 'Equation(derivative(u(t, θ), t), t + 1)' to 'derivative(u, t, 1, 1, θ) - (t + 1)'

2)  Equation: Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

    Take expressions in the form:
     Equation(derivative(derivative(u(x, y, θ), x), x) + derivative(derivative(u(x, y, θ), y), y), -(sin(πx)) * sin(πy))
    to
     (derivative(1, x, y, 1, 2, θ) + derivative(1, x, y, 2, 2, θ)) - -(sin(πx)) * sin(πy)

3)  System of equation: [Dx(u1(x,y,θ)) + 4*Dy(u2(x,y,θ)) ~ 0,
                         Dx(u2(x,y,θ)) + 9*Dy(u1(x,y,θ)) ~ 0]

    Take expressions in the form:
    2-element Array{Equation,1}:
        Equation(derivative(u1(x, y, θ), x) + 4 * derivative(u2(x, y, θ), y), ModelingToolkit.Constant(0))
        Equation(derivative(u2(x, y, θ), x) + 9 * derivative(u1(x, y, θ), y), ModelingToolkit.Constant(0))
    to
      [(derivative(1, x, y, 1, 1, θ) + 4 * derivative(2, x, y, 2, 1, θ)) - 0,
       (derivative(2, x, y, 1, 1, θ) + 9 * derivative(1, x, y, 2, 1, θ)) - 0]

"""
function extract_pde(eqs,_indvars,_depvars,dict_indvars,dict_depvars)
    depvars = [d.name for d in _depvars]
    indvars = Expr.(_indvars)
    vars = :(phi, vec, θ, derivative)
    if !(eqs isa Array)
        eqs = [eqs]
    end
    pde_funcs= []
    for eq in eqs
        _left_expr = (ModelingToolkit.simplified_expr(eq.lhs))
        left_expr = simplified_derivative(_left_expr,indvars,depvars,dict_indvars,dict_depvars)

        _right_expr = (ModelingToolkit.simplified_expr(eq.rhs))
        right_expr = simplified_derivative(_right_expr,indvars,depvars,dict_indvars,dict_depvars)

        pde_func = :($left_expr - $right_expr)
        push!(pde_funcs,pde_func)
    end

    pde_func = build_loss_function(pde_funcs,vars,depvars,indvars,dict_depvars)
    return pde_func
end

# get agruments from bondary condition functions
function get_bc_argument(left_expr)
    if left_expr.args[1] == :derivative
        bcs_arg = left_expr.args[3:end-3]
    else
        bcs_arg = left_expr.args[2:end-1]
    end
    bcs_arg
end

"""
Extract initial and boundary conditions expression to the inner representation.

Examples:

Boundary conditions: [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
                      u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]

Take expression in form:

4-element Array{Equation,1}:
 Equation(u(0, y), ModelingToolkit.Constant(0.0f0))
 Equation(u(1, y), -1.2246467991473532e-16 * sin(πy))
 Equation(u(x, 0), ModelingToolkit.Constant(0.0f0))
 Equation(u(x, 1), -(sin(πx)) * 1.2246467991473532e-16)

to
 [[u(0, y, θ) - 0.0f0],
  [u(1, y, θ) - -1.2246467991473532e-16 * sin(πy)],
  [u(x, 0, θ) - 0.0f0],
  [u(x, 1, θ) - -(sin(πx)) * 1.2246467991473532e-16]]
"""
function extract_bc(bcs,_indvars,_depvars,dict_indvars,dict_depvars)
    depvars = [d.name for d in _depvars]
    indvars = Expr.(_indvars)
    vars = :(phi, vec, θ, derivative)

    bc_funcs = []
    bc_args = []
    _funcs= []
    for _bc in bcs
        if !(_bc isa Array)
            _bc = [_bc]
        end
        _bc_funcs= []
        for bc in _bc
            _left_expr = (ModelingToolkit.simplified_expr(bc.lhs))
            left_expr = simplified_derivative(_left_expr,indvars,depvars,dict_indvars,dict_depvars)

            _right_expr = (ModelingToolkit.simplified_expr(bc.rhs))
            right_expr = simplified_derivative(_right_expr,indvars,depvars,dict_indvars,dict_depvars)

            _bc_func = :($left_expr - $right_expr)
            push!(_bc_funcs,_bc_func)
        end
        bc_func = build_loss_function(_bc_funcs,vars,depvars,indvars,dict_depvars)


        _left_expr = (ModelingToolkit.simplified_expr(_bc[1].lhs))
        left_expr = simplified_derivative(_left_expr,indvars,depvars,dict_indvars,dict_depvars)
        bc_arg = get_bc_argument(left_expr)

        push!(bc_funcs, bc_func)
        push!(bc_args, bc_arg)
    end
    return bc_funcs,bc_args
end

# Generate training set with the points in the domain and on the boundary
function generator_training_sets(domains, discretization, bound_args, dict_indvars)
    dx = discretization.dx

    spans = [d.domain.lower:dx:d.domain.upper for d in domains]
    dict_var_span = Dict([Symbol(d.variables) => d.domain.lower:dx:d.domain.upper for d in domains])

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

    [train_domain_set,train_bound_set]
end

function get_loss_function(pde_func, bc_funcs, train_sets)
    # the points in the domain and on the boundary
    train_domain_set, train_bound_set = train_sets

    # coefficients for loss function
    τb = length(train_bound_set)
    τf = length(train_domain_set)

    # Loss function for pde equation
    function inner_domain_loss(phi, x, θ, derivative)
        sum(pde_func(phi, x, θ, derivative))
    end

    # Loss function for initial and boundary conditions
    function inner_bound_loss(f,phi,x,θ,derivative)
        sum(f(phi, x, θ, derivative))
    end

    function loss_domain(θ, phi, derivative)
        sum(abs2,inner_domain_loss(phi, x, θ, derivative) for x in train_domain_set)
    end

    function loss_boundary(θ, phi, derivative)
       sum(sum(abs2,inner_bound_loss(f,phi,x,θ,derivative) for x in set) for (f,set) in zip(bc_funcs,train_bound_set))
    end

    # General loss function
    function loss(θ, phi, derivative)
        1.0f0/τf * loss_domain(θ, phi, derivative) + 1.0f0/τb * loss_boundary(θ, phi, derivative) #+ 1.0f0/τc * custom_loss(θ)
    end
    return loss
end

# Convert a PDE problem into OptimizationProblem
function DiffEqBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eq = pde_system.eq
    bcs = pde_system.bcs
    domains = pde_system.domain
    indvars = pde_system.indvars
    depvars = pde_system.depvars
    # dictionaries: variable -> unique number
    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)

    dim = length(domains)

    # extract pde
    pde_func = eval(extract_pde(eq,indvars,depvars, dict_indvars,dict_depvars))
    # extract initial and boundary conditions
    extract_bc_funcs, bound_args = extract_bc(bcs,indvars,depvars,dict_indvars,dict_depvars)
    bc_funcs = eval.(extract_bc_funcs)

    # generate training sets
    train_sets = generator_training_sets(domains, discretization, bound_args, dict_indvars)

    # get loss_function
    loss_function = get_loss_function(pde_func,bc_funcs,train_sets)

	return GalacticOptim.OptimizationProblem(loss_function, zeros(dim), p=nothing)
end

function DiffEqBase.solve(
    prob::GalacticOptim.OptimizationProblem,
    alg::NNDE,
    args...;
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)


    loss_function = prob.f

    # dimensionality of equation
    dim = length(prob.x)

    dim > 3 && error("While only dimensionality no more than 3")

    # neural network
    chain  = alg.chain
    # optimizer
    opt    = alg.opt
    # AD flag
    autodiff = alg.autodiff
    # weights of neural network
    initθ = alg.initθ

    # equation/system of equations
    isuinplace = length(chain(zeros(dim),initθ)) == 1

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

    # Calculate derivative
    if autodiff # automatic differentiation (not implemented yet)
        # derivative = (x,θ,n) -> ForwardDiff.gradient(x->phi(x,θ),x)[n]
    else # numerical differentiation
        epsilon = cbrt(eps(Float64))
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

    loss = (θ) -> loss_function(θ, phi, derivative)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    phi ,res
end #solve
