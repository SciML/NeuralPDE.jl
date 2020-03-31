struct NNGeneralPDE{C,O,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    autodiff::Bool
    kwargs::K
end

NNGeneralPDE(chain,opt=Optim.BFGS();autodiff=false,kwargs...) = NNGeneralPDE(chain,opt,autodiff,kwargs)

function DiffEqBase.solve(
    prob::GeneranNNTwoDimPDEProblem,
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
    yspan = prob.yspan
    dt = prob.dt
    dx = prob.dx
    dy = prob.dy
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
    ys = yspan[1]:dy:yspan[2]

    dom_ts = ts[2:end]
    dom_xs = xs[2:end-1]
    dom_ys = ys[2:end-1]

    #init points
    get_points(xs,ys,ts) = [(x,y,t)  for x in xs, y in ys, t in ts]

    boundary_points = get_points(xs,ys,ts[1])
    init_points = [get_points(xs[1],ys,dom_ts), get_points(xs[end],ys,dom_ts),
                   get_points(dom_xs,ys[1],dom_ts), get_points(dom_xs,ys[end],dom_ts)]

    #train sets
    train_bound_set = [(x,y, bound_t,bound_cond)
                      for ((x,y,bound_t),bound_cond) in zip(boundary_points,boundary_conditions)]

    train_initial_set = [[(i_x,i_y,t,init_cond)
                        for ((i_x, i_y, t), init_cond) in zip(init_point,initial_condition)]
                        for (init_point,initial_condition) in zip(init_points,initial_conditions)]

    train_domain_set = get_points(dom_xs,dom_ys,dom_ts)
    # coefficients for loss function
    τb = length(train_bound_set)
    τi = length(train_initial_set)*length(train_initial_set[1])
    τf = length(train_domain_set)

    if chain isa FastChain
        initθ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,y,t,θ) -> first(chain(adapt(typeof(θ),collect([x;y;t])),θ))
        else
            phi = (x,y,t,θ) -> chain(adapt(typeof(θ),collect([x;y;t])),θ)
        end
    else
        initθ,re  = Flux.destructure(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,y,t,θ) -> first(re(θ)(adapt(typeof(θ),collect([x;y;t]))))
        else
            phi = (x,y,t,θ) -> re(θ)(adapt(typeof(θ),collect([x;y;t])))
        end
    end

    if autodiff
        dfdt = (x,y,t,θ) -> ForwardDiff.derivative(x->phi(x,y,t,θ),x)
        dfdx = (x,y,t,θ) -> ForwardDiff.derivative(t->phi(x,y,t,θ),t)
        dfdy = (x,y,t,θ) -> ForwardDiff.derivative(t->phi(x,y,t,θ),y)
        # dfdt = (x,t,θ;xt=[x, t]) -> ForwardDiff.gradient(xt->phi(xt[1],xt[2],θ),xt)[2]
        # dfdx = (x,t,θ;xt=[x, t]) -> ForwardDiff.gradient(xt->phi(xt[1],xt[2],θ),xt)[1]
    else
        dfdt = (x,y,t,θ) -> (phi(x,y,t+cbrt(eps(t)),θ) - phi(x,y,t,θ))/cbrt(eps(t))
        dfdx = (x,y,t,θ) -> (phi(x+cbrt(eps(x)),y,t,θ) - phi(x,y,t,θ))/cbrt(eps(x))
        dfdy = (x,y,t,θ) -> (phi(x,y+cbrt(eps(y)),t,θ) - phi(x,y,t,θ))/cbrt(eps(y))
        epsilon(dv) = cbrt(eps(typeof(dv)))
        #second order central
        dfdtt = (x,y,t,θ) -> (phi(x,y,t+epsilon(dt),θ) - 2phi(x,y,t,θ) + phi(x,y,t-epsilon(dt),θ))/epsilon(dt)^2
        dfdxx = (x,y,t,θ) -> (phi(x+epsilon(dx),y,t,θ) - 2phi(x,y,t,θ) + phi(x-epsilon(dx),y,t,θ))/epsilon(dx)^2
        dfdyy = (x,y,t,θ) -> (phi(x,y+epsilon(dy),t,θ) - 2phi(x,y,t,θ) + phi(x,y-epsilon(dy),t,θ))/epsilon(dy)^2
    end

    #loss function for pde equation
    function inner_loss_domain(x,y,t,θ)
        sum(abs2,pde_func(x,y,t,θ))
    end

    function loss_domain(θ)
        sum(abs2,inner_loss_domain(x,y,t,θ) for (x,y,t) in train_domain_set)
    end

    #Dirichlet boundary
    function inner_loss(x,y,t,θ,cond)
        sum(abs2,phi(x,y,t,θ) - cond)
    end

    #loss function for boundary condiiton
    function loss_boundary(θ)
       sum(abs2,inner_loss(x,y, bound_t,θ,bound_cond)
       for (x,y,bound_t,bound_cond) in train_bound_set)
    end

    #loss function for initial condiiton
    function loss_initial(θ)
        sum(sum(abs2,inner_loss(init_x,init_y,t,θ,init_cond)
        for (init_x,init_y,t,init_cond) in train_init_set) for train_init_set in train_initial_set)
    end

    #loss function for training
    loss(θ) = 1.0f0/τf * loss_domain(θ) + 1.0f0/τb * loss_boundary(θ) + 1.0f0/τi * loss_initial(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    #solutions at timepoints
    if isuinplace
        u = [reshape([first(phi(x,y,t,res.minimizer))
        for x in xs for y in ys],(length(xs),length(ys))) for t in ts]
    else
        u = [reshape([phi(x,y,t,res.minimizer)
        for x in xs for y in ys](length(xs),length(ys))) for t in ts]
    end

    # sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    # DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    # sol
    u, phi ,res
end #solve
