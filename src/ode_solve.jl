struct NNODE{C,O,P,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    initθ::P
    autodiff::Bool
    kwargs::K
end
function NNODE(chain,opt=Optim.BFGS(),init_params = nothing;autodiff=false,kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    NNODE(chain,opt,initθ,autodiff,kwargs)
end

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
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
    autodiff = alg.autodiff

    #train points generation
    ts = tspan[1]:dt:tspan[2]
    initθ = alg.initθ

    if chain isa FastChain
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(chain(adapt(typeof(θ),[t]),θ))
        else
            phi = (t,θ) -> u0 + (t-tspan[1])*chain(adapt(typeof(θ),[t]),θ)
        end
    else
        _,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(re(θ)(adapt(typeof(θ),[t])))
        else
            phi = (t,θ) -> u0 + (t-tspan[1])*re(θ)(adapt(typeof(θ),[t]))
        end
    end

    if autodiff
        dfdx = (t,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
    else
        dfdx = (t,θ) -> (phi(t+sqrt(eps(t)),θ) - phi(t,θ))/sqrt(eps(t))
    end

    function inner_loss(t,θ)
        sum(abs2,dfdx(t,θ) - f(phi(t,θ),p,t))
    end
    function loss(θ)
        include_frac = .50
        sizeof = size(ts)[1]
        total = 0
        for t in 1:round(include_frac*sizeof, digits=0)
            elem = convert(Int64, round(sizeof*rand(1)[1] + 0.5, digits=0))
            total += inner_loss(ts[elem],θ)^2
        end
        return total
    end


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
