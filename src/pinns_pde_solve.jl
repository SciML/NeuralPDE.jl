#TODO generic version with NNODE
struct NNPDE{C,O,P,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    initθ::P
    autodiff::Bool
    kwargs::K
end
function NNPDE(chain,opt=Optim.BFGS(),init_params = nothing;autodiff=false,kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    NNPDE(chain,opt,initθ,autodiff,kwargs)
end

# function _simplified_expr(O::Operation)
#     # for arg in  O.args:
#     if O.op isa Differential
#         return :(derivative($(_simplified_expr(O.args[1])),$(ModelingToolkit.simplified_expr(O.op.x)),$(ModelingToolkit.simplified_expr(O.op.θ))))
#     elseif O.op isa Variable
#         isempty(O.args) && return O.op.name
#         return Expr(:call, Symbol(O.op), simplified_expr.(O.args)...)
#     end
# end

function extract_eq(eq,indvars,depvars)
    vars = :($(Expr.(indvars)...),$([d.name for d in depvars]...), derivative,second_order_derivative)
    left_expr = simplified_expr(eq.lhs)
    right_expr = simplified_expr(eq.rhs)
    _f = eval(:(($vars) -> $left_expr - $right_expr))
    return _f
end

function extract_bc(bcs,indvars,depvars,dim)
    bc_fs = []
    vars = :($(Expr.(indvars[1:end-1])...),)
    for i =1:length(bcs)
        # bcs[1].rhs isa ModelingToolkit.Operation
        if isa(bcs[i].rhs,ModelingToolkit.Operation)
            expr =Expr(bcs[i].rhs)
            _f = eval(:(($vars) -> $expr))
            f = (vars...) -> @eval $_f($vars...)
        # bcs[i].rhs isa ModelingToolkit.Constant
        elseif isa(bcs[i].rhs,ModelingToolkit.Constant)
            f = (vars...) -> bcs[i].rhs.value
        end
        push!(bc_fs, f)
    end
    return bc_fs
end

function DiffEqBase.solve(
    prob::NNPDEProblem,
    alg::NNPDE,
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

    # hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff
    initθ = alg.initθ

    isuinplace = dx isa Number

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
        #TODO support array and tuple of variables
        epsilon(dx) = cbrt(eps(typeof(dx)))
        e = epsilon(dx)
        eps_dx = epsilon(dx)
        eps_masks = [[[e]],
                    [[e,0.0], [0.0,e]],
                    [[e,0.0,0.0], [0.0,e,0.0],[0.0,0.0,e]]]

        dict_indvars = Dict( [string(v) .=> i for (i,v) in enumerate(indvars[1:end-1])])

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
    pde_func = (_x,θ) -> _pde_func(_x...,θ,_u,derivative,second_order_derivative)

    # extract boundary conditions function
    bc_fs = extract_bc(bcs,indvars,depvars,dim)

    # boundary conditions
    dom_spans = [(d.domain.lower:discretization.dxs:d.domain.upper)[2:end-1] for d in domains]
    spans = [d.domain.lower:discretization.dxs:d.domain.upper for d in domains]

    #TODO get more generally points generator avoiding if_else case
    #TODO add residual_points_generator
    get_train_bound() =0
    if dim == 1
        get_train_bound(x,b_f) = ([x],b_f(x))
        xs = spans[1]
        dom_xs = dom_spans[1]
        train_bound_set = [get_train_bound(xs[1],bc_fs[1]);
                           get_train_bound(xs[end],bc_fs[2])]
        train_domain_set = [[x] for x in dom_xs]
    elseif dim == 2
        get_train_bound(xs,ys,b_f) = [([x,y],b_f(x,y)) for x in xs for y in ys]
        xs,ys = spans
        dom_xs,dom_ys = dom_spans

        #square boundary condition
        train_bound_set = [get_train_bound(xs,ys[1],bc_fs[1]);
                           get_train_bound(xs,ys[end],bc_fs[2]);
                           get_train_bound(xs[1],dom_ys,bc_fs[3]);
                           get_train_bound(xs[end],dom_ys,bc_fs[4])]

        train_domain_set = [[x,y]  for x in dom_xs for y in dom_ys]
    else# dim == 3
        get_train_bound(xs,ys,ts,b_f) = [([x,y,t],b_f(x,y,t)) for x in xs for y in ys for t in ts]
        xs,ys,ts = spans
        dom_xs,dom_ys,dom_ts = dom_spans

        train_bound_set = [get_train_bound(xs,ys,ts[1],bc_fs[1]);
                           get_train_bound(xs,ys,ts[end],bc_fs[2]);
                           get_train_bound(xs[1],ys,dom_ts,bc_fs[3]);
                           get_train_bound(xs[end],ys,dom_ts,bc_fs[4]);
                           get_train_bound(dom_xs,ys[1],dom_ts,bc_fs[5]);
                           get_train_bound(dom_xs,ys[end],dom_ts,bc_fs[6])]
        #train sets
        train_domain_set = [[x,y,t]  for x in dom_xs for y in dom_ys for t in dom_ts]
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

    #loss function for training
    loss(θ) = 1.0f0/τf * loss_domain(θ) + 1.0f0/τb * loss_boundary(θ) #+ 1.0f0/τc * custom_loss(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    phi ,res
end #solve
