struct NNGeneralPDE{C,O,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    autodiff::Bool
    kwargs::K
end

NNGeneralPDE(chain,opt=Optim.BFGS();autodiff=false,kwargs...) = NNGeneralPDE(chain,opt,autodiff,kwargs)

# function generate_nn_pde_function(sys::PDESystem, vs = sys.vs, ps = sys.ps, expression = Val{true}; kwargs...)
    ## parse(eq = Dt(u(t,x)) ~ Dx(u(t,x))) -> func(phi, p, x,t)  =  diff(u,y)
    ## f = du/dt - du/dx
    # rhss = [eq.rhs for eq ∈ sys.eqs]
    # lhss = [eq.lhs for eq ∈ sys.eqs]
    # vs′ = [clean(v) for v ∈ vs]
    # ps′ = [clean(p) for p ∈ ps]
    # return build_pde_function(lhssrhss, vs′, ps′, kwargs...)
# function generate_nn_pde_function(pdesys::PDESystem,pde_func, dx=0.1)#, vs = sys.vs, ps = sys.ps, expression = Val{true}; kwargs...)
#     tdomain = pdesys.domain[1].domain
#     domain = pdesys.domain[2].domain
#     @assert domain isa IntervalDomain
#     interior = domain.lower+dx:dx:domain.upper-dx
#     X = domain.lower:dx:domain.upper
#     Q = DirichletBC(0.0,0.0) #BC
        # tspan =(tdomain.lower,tdomain.upper)
#     function f(du,u,p,t)
        # build_pde_function(lhssrhss, vs′, ps′, kwargs...)
#     end
#     u0 = 0.
#     PDEProblem(ODEProblem(nn_pde_func,u0,(tdomain.lower,tdomain.upper),nothing),Q,X)
# end


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


    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    isuinplace = dx isa Number

    #train points generation
    ts = tspan[1]:dt:tspan[2]
    xs = xspan[1]:dx:xspan[2]

    #domain points
    dom_ts = ts[2:end-1]
    #initial points
    bound_ts = [ts[1];ts[end]]

    # coefficients for loss function
    τb = length(boundary_conditions)
    τf = length(ts)*length(xs)

    if chain isa FastChain
        initθ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,t,θ) -> first(chain([x;t],θ))
        else
            phi = (x,t,θ) -> chain([x;t],θ)
        end
    else
        initθ,re  = Flux.destructure(chain)
        #The phi trial solution
        if isuinplace
            phi = (x,t,θ) -> first(re(θ)([x;t]))
        else
            phi = (x,t,θ) -> re(θ)([x;t])
        end
    end

    if autodiff
        dfdt = (x,t,θ) -> ForwardDiff.derivative(x->phi(x,t,θ),x)
        dfdx = (x,t,θ) -> ForwardDiff.derivative(t->phi(x,t,θ),t)
        # dfdt = (x,t,θ;xt=[x, t]) -> ForwardDiff.gradient(xt->phi(xt[1],xt[2],θ),xt)[2]
        # dfdx = (x,t,θ;xt=[x, t]) -> ForwardDiff.gradient(xt->phi(xt[1],xt[2],θ),xt)[1]
    else
        dfdt = (x,t,θ) -> (phi(x,t+sqrt(eps(t)),θ) - phi(x,t,θ))/sqrt(eps(t))
        dfdx = (x,t,θ) -> (phi(x+sqrt(eps(x)),t,θ) - phi(x,t,θ))/sqrt(eps(x))
    end

    #numerical second order
    dfdtt= (x,t,θ) -> (phi(x,t+epsilon(t),θ) .- 2phi(x,t,θ) .+ phi(x,t-epsilon(t),θ))/epsilon(t)^2
    dfdxx = (x,t,θ) -> (phi(x+epsilon(x),t,θ) .- 2phi(x,t,θ) .+ phi(x-epsilon(x),t,θ))/epsilon(x)^2

    #loss function for pde equation
    function inner_loss_domain(x,t,θ)
        sum(abs2,pde_func(x,t,θ))
    end

    function loss_domain(θ)
        sum(abs2,inner_loss_domain(x,t,θ) for x in xs for t in ts)
    end

    #loss function for boundary condiiton
    #Dirichlet boundary
    function inner_loss_boundary(x,t,θ,bc)
        sum(abs2, phi(x,t,θ) - bc)
    end
    function loss_boundary(θ)
       loss_one_boundary(bound_t,bound_cond) = sum(abs2,inner_loss_boundary(x, bound_t, θ, bc) for (x, bc) in zip(xs,bound_cond))
       sum(abs2,loss_one_boundary(bound_t,bound_cond) for (bound_t, bound_cond) in zip(bound_ts,boundary_conditions))
    end

    #loss function for training
    loss(θ) = 1/τb * loss_boundary(θ) + 1/τf * loss_domain(θ)

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
    u
end #solve
