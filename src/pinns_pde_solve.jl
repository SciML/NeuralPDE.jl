struct PhysicsInformedNN{int}
  dx::int
end

"""
Create dictionary: variable => unique number for variable

# Example

Dict{Symbol,Int64} with 3 entries:
  :y => 2
  :t => 3
  :x => 1
"""
get_dict_indvars(indvars) = Dict( [Symbol(v) .=> i for (i,v) in enumerate(indvars)])

# Calculate an order of derivative
function count_order(_args)
    n = 0
    while (_args[1] == :derivative)
        n = n+1
        _args = _args[2].args
    end
    return n
end

# Wrapper for _simplified_derivative
function simplified_derivative(ex,indvars,depvars,dict_indvars)
    if ex isa Expr
        ex.args = _simplified_derivative(ex.args,indvars,depvars,dict_indvars)
    end
    return ex
end

"""
Simplify the derivative expression

# Examples

1. First derivative of function 'u' of one variable: 'x'

Take expressions in the form: `derivative(u(x,θ), x)` to `derivative(u, x, unn, order, θ)`,

where
 'u' - the variable
 'x' - coordinates of point
 'unn' - unique number for the variable
 'order' - order of derivative
 'θ' - parameters of neural network

"""
function _simplified_derivative(_args,indvars,depvars,dict_indvars)
    for (i,e) in enumerate(_args)
        if !(e isa Expr)
            # if autodiff
            #     error("While autodiff not support")
            # else
            # end
            if e == :derivative && _args[end] != :θ
                order = count_order(_args)
                _args = [:derivative,depvars..., indvars...,dict_indvars[_args[3]], order, :θ]
                break
            end
        else
            _args[i].args = _simplified_derivative(_args[i].args,indvars,depvars,dict_indvars)
        end
    end
    return _args
end

"""
Extract ModelingToolkit PDE form to the inner representation.

Example:

1)  Equation: Dt(u(t,θ)) ~ t +1

    Take expressions in the form: 'Equation(derivative(u(t, θ), t), t + 1)' to 'derivative(u, t, 1, 1, θ) - (t + 1)'

2)  Equation: Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

    Take expressions in the form:
    'Equation(derivative(derivative(u(x, y, θ), x), x) + derivative(derivative(u(x, y, θ), y), y), -(sin(πx)) * sin(πy))'
    to
    '(derivative(u, x, y, 1,,2, θ) + derivative(u, x, y, 2, 2, θ)) - -(sin(πx)) * sin(πy)'

"""
function extract_eq(eq,_indvars,_depvars,dict_indvars)
    depvars = [d.name for d in _depvars]
    indvars = Expr.(_indvars)
    vars = :($(depvars...), $(indvars...), θ, derivative)

    _left_expr = (ModelingToolkit.simplified_expr(eq.lhs))
    left_expr = simplified_derivative(_left_expr,indvars,depvars,dict_indvars)

    _right_expr = (ModelingToolkit.simplified_expr(eq.rhs))
    right_expr = simplified_derivative(_right_expr,indvars,depvars,dict_indvars)

    _f = eval(:(($vars) -> $left_expr - $right_expr))
    return _f
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

4-element Array{Any,1}:
 (0, :x, '_f(x,y) = 0.0f0')
 (1, :x, '_f(x,y) = -1.22e-16 * sin(πy)')
 (0, :y, '_f(x,y) = 0.0f0')
 (1, :y, '_f(x,y) = -(sin(πx)) * 1.22e-16')

"""

function extract_bc(bcs,indvars,depvars)
    output= []
    vars = Expr.(indvars)
    vars_expr = :($(Expr.(indvars)...),)
    for i =1:length(bcs)
        bcs_args = simplified_expr(bcs[i].lhs.args)
        bc_point_var = first(filter(x -> !(x in bcs_args), vars))
        bc_point = first(bcs_args[typeof.(bcs_args) .!= Symbol])
        if isa(bcs[i].rhs,ModelingToolkit.Operation)
            expr =Expr(bcs[i].rhs)
            _f = eval(:(($vars_expr) -> $expr))
            f = (vars_expr...) -> @eval $_f($vars_expr...)
        elseif isa(bcs[i].rhs,ModelingToolkit.Constant)
            f = (vars_expr...) -> bcs[i].rhs.value
        end
        push!(output, (bc_point,bc_point_var,f))
    end
    return output
end

# Generate training set with the points in the domain and on the boundary
function generator_training_sets(domains, discretization, bound_data, dict_indvars)
    dx = discretization.dx

    dom_spans = [(d.domain.lower:dx:d.domain.upper)[2:end-1] for d in domains]
    spans = [d.domain.lower:dx:d.domain.upper for d in domains]

    train_set = []
    for points in Iterators.product(spans...)
        push!(train_set, [points...])
    end

    train_domain_set = []
    for points in Iterators.product(dom_spans...)
        push!(train_domain_set, [points...])
    end

    train_bound_set = []
    for (point,symb,bc_f) in bound_data
        b_set = [(points, bc_f(points...)) for points in train_set if points[dict_indvars[symb]] == point]
        push!(train_bound_set, b_set...)
    end

    [train_domain_set,train_bound_set]
end

# Convert a PDE problem into an PINNs problem
function DiffEqBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eq =pde_system.eq
    bcs = pde_system.bcs
    domains = pde_system.domain
    indvars = pde_system.indvars
    depvars = pde_system.depvars
    dx = discretization.dx
    dim = length(domains)

    # dictionary: variable -> unique number
    dict_indvars = get_dict_indvars(indvars)

    # extract equation
    _pde_func = extract_eq(eq,indvars,depvars, dict_indvars)

    # extract initial and boundary conditions
    bound_data  = extract_bc(bcs,indvars,depvars)

    # generate training sets
    train_sets = generator_training_sets(domains, discretization, bound_data, dict_indvars)

    return NNPDEProblem(_pde_func, train_sets, dim)
end

function DiffEqBase.solve(
    prob::NNPDEProblem,
    alg::NNDE,
    args...;
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    # DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    # pde function
    _pde_func = prob.pde_func
    # training sets
    train_sets = prob.train_sets
    # the points in the domain and on the boundary
    train_domain_set, train_bound_set = train_sets

    # coefficients for loss function
    τb = length(train_bound_set)
    τf = length(train_domain_set)

    # dimensionality of equation
    dim = prob.dim

    dim > 3 && error("While only dimensionality no more than 3")

    # neural network
    chain  = alg.chain
    # optimizer
    opt    = alg.opt
    # AD flag
    autodiff = alg.autodiff
    # weights of neural network
    initθ = alg.initθ

    # equation/system of equations (not yet supported)
    isuinplace = true

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
            u_in = _x[1]
            x = collect(_x[2:end-3])
            der_num =_x[end-2]
            order = _x[end-1]
            θ = _x[end]
            ε = εs[der_num]
            return _derivative(x,θ,order,ε)
        end
        _derivative = (x,θ,order,ε) ->
        begin
            if order == 1
                #Five-point stencil
                # return (-phi(x+2ε,θ) + 8phi(x+ε,θ) - 8phi(x-ε,θ) + phi(x-2ε,θ))/(12*epsilon)
                return (phi(x+ε,θ) - phi(x-ε,θ))/(2*epsilon)
            else
                return (_derivative(x+ε,θ,order-1,ε) - _derivative(x-ε,θ,order-1,ε))/(2*epsilon)
            end
        end
    end
    # Represent pde function
    _u= (_x...; x = collect(_x[1:end-1]), θ = _x[end]) -> phi(x,θ)
    pde_func = (_x,θ) -> _pde_func(_u,_x...,θ, derivative)


    # Loss function for pde equation
    function inner_loss_domain(x,θ)
        pde_func(x,θ)
    end

    function loss_domain(θ)
        sum(abs2,inner_loss_domain(x,θ) for x in train_domain_set)
    end

    # Loss function for Dirichlet boundary
    function inner_loss(x,θ,bound)
        phi(x,θ) - bound
    end

    # # Neumann boundary
    # function inner_neumann_loss(x,θ,num,order,bound)
    #     derivative(u,x,der_num,order, θ) - bound
    # end

    # Loss function for initial and boundary conditions
    function loss_boundary(θ)
       sum(abs2,inner_loss(x,θ,bound) for (x,bound) in train_bound_set)
    end

    # General loss function
    loss(θ) = 1.0f0/τf * loss_domain(θ) + 1.0f0/τb * loss_boundary(θ) #+ 1.0f0/τc * custom_loss(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    phi ,res
end #solve
