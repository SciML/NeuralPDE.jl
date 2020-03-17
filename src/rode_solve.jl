struct NNRODE{C,W,O,P,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    W::W
    opt::O
    initθ::P
    kwargs::K
end
function NNRODE(chain,W,opt=Optim.BFGS(),init_params = nothing;kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    NNRODE(chain,W,opt,initθ,kwargs)
end

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractRODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    Wg = alg.W
    #train points generation
    ts = tspan[1]:dt:tspan[2]
    initθ = alg.initθ

    if chain isa FastChain
        #The phi trial solution
        if u0 isa Number
            phi = (t,W,θ) -> u0 + (t-tspan[1])*first(chain(adapt(typeof(θ),collect([t,W])),θ))
        else
            phi = (t,W,θ) -> u0 + (t-tspan[1])*chain(adapt(typeof(θ),collect([t,W])),θ)
        end
    else
        _,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,W,θ) -> u0 + (t-tspan[1])*first(re(θ)(adapt(typeof(θ),collect([t,W]))))
        else
            phi = (t,W,θ) -> u0 + (t-tspan[1])*re(θ)(adapt(typeof(θ),collect([t,W])))
        end
    end

    dfdx = (x,θ) -> ForwardDiff.gradient((x)->phi(x[1],x[2],θ),x)[1]
    # dfdx = (t,W,θ) -> Zygote.gradient((t,W,θ)->phi(t,W,θ),t,W,θ)[1]

    function inner_loss(t,W,θ)
        x = [t,W]
        sum(abs2,dfdx(x,θ) - f(phi(t,W,θ),p,W,t))
    end
    loss(θ) = sum(abs2,inner_loss(t,W,θ) for (t , W) in zip(ts , Wg.u) )
    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    solutions at timepoints
    if u0 isa Number
        u = [first(phi(t,W.u,res.minimizer)) for (t , W) in zip(ts , Wg)]
    else
        u = [phi(t,W,res.minimizer) for (t,W) in zip(ts,Wg.u)]
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
