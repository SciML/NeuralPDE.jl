struct NNGenODE{C,IC,O,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    init_cond::IC
    opt::O
    autodiff::Bool
    kwargs::K
end

NNGenODE(chain,init_cond,opt=Optim.BFGS();autodiff=false,kwargs...) = NNGenODE(chain,init_cond,opt,autodiff,kwargs)

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NNGenODE,
    args...;
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    #initial condition
    init_cond = alg.init_cond

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    #train points generation
    ts = tspan[1]:dt:tspan[2]
    #domain points
    dom_ts = ts[2:end-1]
    #initial points
    inti_ts = [ts[1];ts[end]]

    # coefficients for loss function
    τi = length(inti_ts)
    τf = length(dom_ts)

    if chain isa FastChain
        initθ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> first(chain([t],θ))
        else
            phi = (t,θ) -> chain([t],θ)
        end
    else
        initθ,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> first(re(θ)([t]))
        else
            phi = (t,θ) -> re(θ)([t])
        end
    end

    if autodiff
        dfdx = (t,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
    else
        dfdx = (t,θ) -> (phi(t+sqrt(eps(t)),θ) - phi(t,θ))/sqrt(eps(t))
    end

    #=
    square_square(x) = x.^2
    function loss(θ)
        _dfdx_diff = square_square.(dfdx.(ts,(θ,)) .- f.(phi.(ts,(θ,)),(p,),ts))
        sum(sum(_dfdx_diff))
    end
    =#

    function inner_loss_domain(t,θ)
        sum(abs2,dfdx(t,θ) - f(phi(t,θ),p,t))
    end
    function inner_loss_initial(t,θ,ic)
        sum(abs2, phi(t,θ) - ic)
    end
    #loss function for equation
    loss_domain(θ) = sum(abs2,inner_loss_domain(t,θ) for t in dom_ts)
    #loss function for initial condiiton
    loss_initial(θ) = sum(abs2,inner_loss_initial(t,θ,ic) for (t, ic) in zip(inti_ts,init_cond))

    #loss function for training
    loss(θ) = 1/τi * loss_initial(θ) + 1/τf * loss_domain(θ)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    #solutions at timepoints
    if u0 isa Number
        u = [first(phi(t,res.minimizer)) for t in ts]
    else
        u = [phi(t,res.minimizer) for t in ts]
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
