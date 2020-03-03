struct NNODE{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    autodiff::Bool
end
NNODE(chain,opt=Flux.ADAM(0.1);autodiff=false) = NNODE(chain,opt,autodiff)

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
        θ = DiffEqFlux.initial_params(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t - tspan[1])*first(chain([t],θ))
        else
            phi = (t,θ) -> u0 + (t - tspan[1])*chain([t],θ)
        end
    else
        θ,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t - tspan[1])*first(re(θ)([t]))
        else
            phi = (t,θ) -> u0 + (t - tspan[1])*re(θ)([t])
        end
    end

    if autodiff
        dfdx = (t,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
    else
        dfdx = (t,θ) -> (phi(t+sqrt(eps(t)),θ) - phi(t,θ))/(sqrt(eps(t)))
    end

    loss(θ) = sum(abs2,sum(abs2,dfdx(t,θ) - f(phi(t,θ),p,t)) for t in ts)

    cb = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, θ, opt; cb = cb, maxiters=maxiters)

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
