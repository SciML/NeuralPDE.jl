struct NNPDE{C,O,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    autodiff::Bool
    kwargs::K
end

NNPDE(chain,opt=Optim.BFGS();autodiff=false,kwargs...) = NNPDE(chain,opt,autodiff,kwargs)

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

function DiffEqBase.solve(
    prob::GeneralNNPDEProblem,
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
    bound_funcs = prob.bound_funcs
    domains =  prob.space.domains
    discretization =  prob.space.discretization
    dx = discretization.dxs
    dim = prob.dim
    p = prob.p

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    isuinplace = dx isa Number

    dom_spans = [(d.domain.lower:discretization.dxs:d.domain.upper)[2:end-1] for d in domains]
    spans = [d.domain.lower:discretization.dxs:d.domain.upper for d in domains]

    #TODO get more generally points generator avoiding if_else case
    #TODO add residual_points_generator
    get_train_bound_set() = nothing
    if dim == 1
        get_train_bound_set(xs,bound_funcs) = [([x],bound_funcs(x))  for x in xs]
        xs = spans[1]
        dom_xs = dom_spans[1]
        train_bound_set = [([xs[1]], bound_funcs(xs[1])), ([xs[2]], bound_funcs(xs[2]))]
        train_domain_set = [[x]  for x in dom_xs]
    elseif dim == 2
        get_train_bound_set(xs,ys,bound_funcs) = [([x,y],bound_funcs(x,y))  for x in xs for y in ys]
        xs,ys = spans
        dom_xs,dom_ys = dom_spans
        #square boundary condition
        train_bound_set = [get_train_bound_set(xs,ys[1],bound_funcs); get_train_bound_set(xs,ys[end],bound_funcs);
                           get_train_bound_set(xs[1],dom_ys,bound_funcs); get_train_bound_set(xs[end],dom_ys,bound_funcs)]

        train_domain_set = [[x,y]  for x in dom_xs for y in dom_ys]
    else # dim == 3
        get_train_bound_set(xs,ys,ts,bound_funcs) = [([x,y,t],bound_funcs(x,y,t)) for x in xs for y in ys for t in ts]
        xs,ys,ts = spans
        dom_xs,dom_ys,dom_ts = dom_spans

        train_bound_set = [get_train_bound_set(xs,ys,ts[1],bound_funcs);
                           get_train_bound_set(xs,ys,ts[end],bound_funcs);
                           get_train_bound_set(xs[1],ys,dom_ts,bound_funcs);
                           get_train_bound_set(xs[end],ys,dom_ts,bound_funcs);
                           get_train_bound_set(dom_xs,ys[1],dom_ts,bound_funcs);
                           get_train_bound_set(dom_xs,ys[end],dom_ts,bound_funcs)]
        #train sets
        train_domain_set = [[x,y,t]  for x in dom_xs for y in dom_ys for t in dom_ts]
    end

    # coefficients for loss function
    τb = length(train_bound_set)
    τf = length(train_domain_set)

    if chain isa FastChain
        initθ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,θ) -> first(chain(adapt(typeof(θ),x),θ))
        else
            phi = (x,θ) -> chain(adapt(typeof(θ),x),θ)
        end
    else
        initθ,re  = Flux.destructure(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,θ) -> first(re(θ)(adapt(typeof(θ),x)))
        else
            phi = (x,θ) -> re(θ)(adapt(typeof(θ),x))
        end
    end

    #TODO find another way avoid Mutating arrays
    epsilon(dx) = cbrt(eps(typeof(dx)))
    e = epsilon(dx)
    eps_masks = [
             [[e]],
             [[e, 0.0], [0.0,e]],
             [[e,0.0,0.0], [0.0,e,0.0],[0.0,0.0,e]]
             ]

    if autodiff
        u = (x,θ) -> phi(x,θ)
        du = (x,θ,n) -> ForwardDiff.gradient(x->phi(x,θ),x)[n]
        # du2 = ...
    else
        u = (x,θ) -> phi(x,θ)
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
