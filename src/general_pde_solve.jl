struct NNGeneralPDE{C,O,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    autodiff::Bool
    kwargs::K
end

NNGeneralPDE(chain,opt=Optim.BFGS();autodiff=false,kwargs...) = NNGeneralPDE(chain,opt,autodiff,kwargs)

function DiffEqBase.solve(
    prob::GeneranNNPDEProblem,
    alg::NNGeneralPDE,
    args...;
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    # DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")


    tspan = prob.tspan
    xspan = prob.xspan
    dt = prob.dt
    dx = prob.dx
    pde_func = prob.pde_func
    p = prob.p
    boundary_conditions = prob.boundary_conditions
    initial_conditions = prob.initial_conditions

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    isuinplace = dx isa Number

    #train points generation
    ts = tspan[1]:dt:tspan[2]
    xs = xspan[1]:dx:xspan[2]

    dom_ts = ts[2:end-1]
    dom_xs = xs[2:end-1]

    #boundary points
    bound_ts = [fill(ts[1],length(xs)), fill(ts[end],length(xs))]
    #initial points
    init_xs = [fill(xs[1],length(ts)), fill(xs[end],length(ts))]

    get_bound_points(xs,ys,ts) = [(x,y,t)  for x in xs, y in ys, t in ts]

    #train sets
    train_bound_set1 = [(x, bound_t,bound_cond) for (x,bound_t,bound_cond) in zip(xs,bound_ts[1],boundary_conditions[1])]
    train_bound_set2 = [(x, bound_t,bound_cond) for (x,bound_t,bound_cond) in zip(xs,bound_ts[2],boundary_conditions[2])]
    train_initial_set1 = [(init_x,t,init_cond) for (init_x,t,init_cond)  in zip(init_xs[1], ts, initial_conditions[1])][2:end-1]
    train_initial_set2 = [(init_x,t,init_cond) for (init_x,t,init_cond)  in zip(init_xs[2], ts, initial_conditions[2])][2:end-1]
    train_bound_set = [train_bound_set1; train_bound_set2; train_initial_set1; train_initial_set2]

    train_domain_set = [(x,t) for x in dom_xs for t in dom_ts]

    # coefficients for loss function
    τb = length(train_bound_set)
    τf = length(train_domain_set)

    if chain isa FastChain
        initθ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,t,θ) -> first(chain(adapt(typeof(θ),collect([x;t])),θ))
        else
            phi = (x,t,θ) -> chain(adapt(typeof(θ),collect([x;t])),θ)
        end
    else
        initθ,re  = Flux.destructure(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,t,θ) -> first(re(θ)(adapt(typeof(θ),collect([x;t]))))
        else
            phi = (x,t,θ) -> re(θ)(adapt(typeof(θ),collect([x;t])))
        end
    end

    if autodiff
        dfdt = (x,t,θ) -> ForwardDiff.derivative(x->phi(x,t,θ),x)
        dfdx = (x,t,θ) -> ForwardDiff.derivative(t->phi(x,t,θ),t)
        # dfdt = (x,t,θ;xt=[x, t]) -> ForwardDiff.gradient(xt->phi(xt[1],xt[2],θ),xt)[2]
        # dfdx = (x,t,θ;xt=[x, t]) -> ForwardDiff.gradient(xt->phi(xt[1],xt[2],θ),xt)[1]
    else
        dfdt = (x,t,θ) -> (phi(x,t+cbrt(eps(t)),θ) - phi(x,t,θ))/cbrt(eps(t))
        dfdx = (x,t,θ) -> (phi(x+cbrt(eps(x)),t,θ) - phi(x,t,θ))/cbrt(eps(x))
        epsilon(dv) = cbrt(eps(typeof(dv)))
        #second order central
        dfdtt = (x,t,θ) -> (phi(x,t+epsilon(dt),θ) - 2phi(x,t,θ) + phi(x,t-epsilon(dt),θ))/epsilon(dt)^2
        dfdxx = (x,t,θ) -> (phi(x+epsilon(dx),t,θ) - 2phi(x,t,θ) + phi(x-epsilon(dx),t,θ))/epsilon(dx)^2
    end

    #loss function for pde equation
    function inner_loss_domain(x,t,θ)
        pde_func(x,t,θ)
    end
    # rxs = collect(rand(xs[1]:dx:xs[end],length(xs)))
    # rts = collect(rand(ts[1]:dt:ts[end],length(ts)))

    function loss_domain(θ)
        sum(abs2,inner_loss_domain(x,t,θ) for x in dom_xs, t in dom_ts)
    end

    #Dirichlet boundary
    function inner_loss(x,t,θ,cond)
        phi(x,t,θ) - cond
    end
    #loss function for boundary condiiton
    function loss_boundary(θ)
       sum(abs2,inner_loss(x,t,θ,bound_cond) for (x,t,bound_cond) in train_bound_set)
    end

    #loss function for training
    loss(θ) = 1.0f0/τf * loss_domain(θ) + 1.0f0/τb * loss_boundary(θ) #+ 1.0f0/τi * loss_initial(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    #solutions at timepoints
    if isuinplace
        u = [[first(phi(x,t,res.minimizer)) for x in xs ] for t in ts ]
    else
        u = [[phi(x,t,res.minimizer)  for x in xs] for t in ts]
    end

    # sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    # DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    # sol
    u, phi ,res
end #solve
