struct NNODE{C,O,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    autodiff::Bool
    kwargs::K
end
NNODE(chain,opt=Optim.BFGS();autodiff=false,kwargs...) = NNODE(chain,opt,autodiff,kwargs)

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

    if chain isa FastChain
        initθ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(chain([t],θ))
        else
            phi = (t,θ) -> u0 + (t-tspan[1])*chain([t],θ)
        end
    else
        initθ,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(re(θ)([t]))
        else
            phi = (t,θ) -> u0 + (t-tspan[1])*re(θ)([t])
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

    function inner_loss(t,θ)
        sum(abs2,dfdx(t,θ) - f(phi(t,θ),p,t))
    end
    loss(θ) = sum(abs2,inner_loss(t,θ) for t in ts) # sum(abs2,phi(tspan[1],θ) - u0)

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
