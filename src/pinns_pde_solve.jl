function extract_eq(eq,indvars,depvars)
    vars = :($([d.name for d in depvars]...), $(Expr.(indvars)...), θ, derivative,second_order_derivative)
    left_expr = ModelingToolkit.simplified_expr(eq.lhs)
    right_expr = ModelingToolkit.simplified_expr(eq.rhs)
    _f = eval(:(($vars) -> $left_expr - $right_expr))
    return _f
end

function extract_bc(bcs,indvars,depvars)
    output= []
    vars = Expr.(indvars)
    vars_expr = :($(Expr.(indvars)...),)
    for i =1:length(bcs)
        bcs_args = simplified_expr(bcs[i].lhs.args)
        bc_vars = bcs_args[typeof.(bcs_args) .== Symbol]
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
    pde_system = prob.pde_system
    eq =pde_system.eq
    bcs = pde_system.bcs
    domains = pde_system.domain
    indvars = pde_system.indvars
    depvars = pde_system.depvars
    discretization =  prob.discretization
    dx = discretization.dxs
    dim = discretization.order
    p = prob.p

    dim > 3 && error("While only dimensionality no more than 3")

    # hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff
    initθ = alg.initθ

    isuinplace = dx isa Number

    dict_indvars = Dict( [Symbol(v) .=> i for (i,v) in enumerate(indvars)])

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

    # derivative
    if autodiff
        # derivative = (x,θ,n) -> ForwardDiff.gradient(x->phi(x,θ),x)[n]
        # second_order_derivative = ...
    else
        #TODO find another way avoid Mutating arrays
        epsilon(dx) = cbrt(eps(typeof(dx)))
        e = epsilon(dx)
        eps_dx = epsilon(dx)
        eps_masks = [[[e]],
                    [[e,0.0], [0.0,e]],
                    [[e,0.0,0.0], [0.0,e,0.0],[0.0,0.0,e]]]

        derivative = (_x...) ->
        begin
            u_in = _x[1]
            x = collect(_x[2:end-2])
            der_num =_x[end-1]
            θ = _x[end]
            # n = get(dict_indvars,der_var,nothing)
            (phi(x+eps_masks[dim][der_num], θ) - phi(x,θ))/eps_dx
        end
        second_order_derivative = (_x...) ->
        begin
            u_in = _x[1]
            x = collect(_x[2:end-2])
            der_num =_x[end-1]
            θ = _x[end]
            # n = get(dict_indvars,der_var,nothing)
            (phi(x+eps_masks[dim][der_num], θ) - 2*phi(x,θ)+ phi(x-eps_masks[dim][der_num], θ) )/(eps_dx^2)
        end
    end

    # pde equation
    _u= (_x...; x = collect(_x[1:end-1]),θ = _x[end]) -> phi(x,θ)
    _pde_func = extract_eq(eq,indvars,depvars)
    pde_func = (_x,θ) -> _pde_func(_u,_x...,θ,derivative,second_order_derivative)

    # extract boundary conditions
    bound_data  = extract_bc(bcs,indvars,depvars)

    # generate training sets
    dom_spans = [(d.domain.lower:discretization.dxs:d.domain.upper)[2:end-1] for d in domains]
    spans = [d.domain.lower:discretization.dxs:d.domain.upper for d in domains]

    #TODO add residual points generator
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

    # coefficients for loss function
    τb = length(train_bound_set)
    τf = length(train_domain_set)


    #loss function for pde equation
    function inner_loss_domain(x,θ)
        pde_func(x,θ)
    end

    function loss_domain(θ)
        sum(abs2,inner_loss_domain(x,θ) for x in train_domain_set)
    end

    #Dirichlet boundary
    function inner_loss(x,θ,bound)
        phi(x,θ) - bound
    end

    # #Neumann boundary
    # function inner_neumann_loss(x,θ,num,bound)
    #     derivative(u,x,num θ) - bound
    # end

    #loss function for boundary condiiton
    function loss_boundary(θ)
       sum(abs2,inner_loss(x,θ,bound) for (x,bound) in train_bound_set)
    end

    #loss function
    loss(θ) = 1.0f0/τf * loss_domain(θ) + 1.0f0/τb * loss_boundary(θ) #+ 1.0f0/τc * custom_loss(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    phi ,res
end #solve
