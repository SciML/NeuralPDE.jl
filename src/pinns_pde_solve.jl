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


# TODO overload struct for high dim case
struct Spaces{DIS}
    domains::Array
    discretization::DIS
end
Spaces(domains,discretization=Discretization(dxs=0.1)) = Spaces(domains,discretization)

# TODO overload struct for high dim case
struct Discretization{}
    dxs::Float64
end
Discretization(dxs=0.1) = Discretization(dxs)

# function extract_eq(bcs,dim)
#     ...
# end

function extract_bc(bcs,dim)
    bc_fs = []
    # TODO extract args form bcs
    args =  dim == 1 ? :(t)  : (dim ==2 ? :(x,y) : :(x,y,t))
    for i =1:length(bcs)
        if isa(bcs[i].rhs,ModelingToolkit.Operation)
            expr =Expr(bcs[i].rhs)
            _f = eval(:(($args) -> $expr))
            f = (args...) -> @eval $_f($args...)
        elseif isa(bcs[i].rhs,ModelingToolkit.Constant)
            f = (args...) -> bcs[i].rhs.value
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
    pde_func = prob.pde_func
    bcs = prob.bound_conditions
    domains =  prob.space.domains
    discretization =  prob.space.discretization
    dx = discretization.dxs
    dim = prob.dim
    p = prob.p

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff
    initθ = alg.initθ

    isuinplace = dx isa Number

    dom_spans = [(d.domain.lower:discretization.dxs:d.domain.upper)[2:end-1] for d in domains]
    spans = [d.domain.lower:discretization.dxs:d.domain.upper for d in domains]

    bc_fs = extract_bc(bcs,dim)
    get_train_bound_set(xs,ys,ts,b_f) = [([x,y,t],b_f(x,y,t)) for x in xs for y in ys for t in ts]
    #TODO get more generally points generator avoiding if_else case
    #TODO add residual_points_generator
    if dim == 1
        xs = spans[1]
        dom_xs = dom_spans[1]
        train_bound_set = [([xs[1]], bc_fs[1](xs[1])), ([xs[end]], bc_fs[2](xs[end]))]
        train_domain_set = [[x]  for x in dom_xs]
    elseif dim == 2
        get_train_bound_set(xs,ys,b_f) = [([x,y],b_f(x,y)) for x in xs for y in ys]
        xs,ys = spans
        dom_xs,dom_ys = dom_spans
        #square boundary condition
        #TODO   list =[(xs,ys[1],bc_fs[1]),(xs,ys[end],bc_fs[2]),...]
        # train_bound_set = [get_train_bound_set(l) for l in list]
        train_bound_set = [get_train_bound_set(xs,ys[1],bc_fs[1]);
                           get_train_bound_set(xs,ys[end],bc_fs[2]);
                           get_train_bound_set(xs[1],dom_ys,bc_fs[3]);
                           get_train_bound_set(xs[end],dom_ys,bc_fs[4])]

        train_domain_set = [[x,y]  for x in dom_xs for y in dom_ys]
    elseif dim == 3
        get_train_bound_set(xs,ys,ts,b_f) = [([x,y,t],b_f(x,y,t)) for x in xs for y in ys for t in ts]
        xs,ys,ts = spans
        dom_xs,dom_ys,dom_ts = dom_spans

        train_bound_set = [get_train_bound_set(xs,ys,ts[1],bc_fs[1]);
                           get_train_bound_set(xs,ys,ts[end],bc_fs[2]);
                           get_train_bound_set(xs[1],ys,dom_ts,bc_fs[3]);
                           get_train_bound_set(xs[end],ys,dom_ts,bc_fs[4]);
                           get_train_bound_set(dom_xs,ys[1],dom_ts,bc_fs[5]);
                           get_train_bound_set(dom_xs,ys[end],dom_ts,bc_fs[6])]
        #train sets
        train_domain_set = [[x,y,t]  for x in dom_xs for y in dom_ys for t in dom_ts]
    end

    # coefficients for loss function
    τb = length(train_bound_set)
    τf = length(train_domain_set)

    if chain isa FastChain
        #The phi trial solution
        if isuinplace
            phi = (x,θ) -> first(chain(adapt(typeof(θ),x),θ))
        else
            phi = (x,θ) -> chain(adapt(typeof(θ),x),θ)
        end
    else
        _,re  = Flux.destructure(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,θ) -> first(re(θ)(adapt(typeof(θ),x)))
        else
            phi = (x,θ) -> re(θ)(adapt(typeof(θ),x))
        end
    end
    # try
    #     phi(train_domain_set[1] , initθ)
    # catch err
    #     if isa(err , DimensionMismatch)
    #         throw( throw(DimensionMismatch("Dimensions of the initial u0 and chain should match")))
    #     else
    #         throw(err)
    #     end
    # end

    if autodiff
        # uf = (x,θ) -> phi(x,θ)
        du = (x,θ,n) -> ForwardDiff.gradient(x->phi(x,θ),x)[n]
        # du2 = ...
    else
        #TODO find another way avoid Mutating arrays
        epsilon(dx) = cbrt(eps(typeof(dx)))
        e = epsilon(dx)
        eps_masks = [
                 [[e]],
                 [[e, 0.0], [0.0,e]],
                 [[e,0.0,0.0], [0.0,e,0.0],[0.0,0.0,e]]
                 ]
        # uf = (x,θ) -> phi(x,θ)
        du = (x,θ,n) -> (phi(collect(x+eps_masks[dim][n]),θ) - phi(x,θ))/epsilon(dx)
        du2 = (x,θ,n) -> (phi(x+eps_masks[dim][n],θ) - 2phi(x,θ) + phi(x-eps_masks[dim][n],θ))/epsilon(dx)^2
    end

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
    # function inner_neumann_loss(x,θ,bound)
    #     du(x,θ,n) - bound
    # end

    #loss function for boundary condiiton
    function loss_boundary(θ)
       sum(abs2,inner_loss(x,θ,bound) for (x,bound) in train_bound_set)
    end

    # function custom_loss(θ)
    #     sum(...)
    # end

    #loss function for training
    loss(θ) = 1.0f0/τf * loss_domain(θ) + 1.0f0/τb * loss_boundary(θ) #+ 1.0f0/τi * custom_loss(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    phi ,res
end #solve
