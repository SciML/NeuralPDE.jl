struct NNRODE{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
end
NNRODE(chain;opt=Flux.ADAM(0.1)) = NNRODE(chain,opt)

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
    ps     = Flux.params(chain)
    data   = Iterators.repeated((), maxiters)

    #train points generation
    ts = tspan[1]:dt:tspan[2]

    #The phi trial solution
    ϕ(t,W) = u0 .- (t.-tspan[1]).*v(Tracker.collect([t,W]))
    
    dfdx = (t,W) -> Tracker.gradient((t,W) -> sum(ϕ(t,W)), t,W; nest = true)[1]
    loss = () -> sum(abs2,sum(abs2,dfdx(t,W).-f(ϕ(t,W)[1],p,t,W)[1]) for (t,W) in zip(ts,randn(length(ts))))

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end
    Flux.train!(loss, ps, data, opt; cb = cb)

    u = [ϕ(t,W)[1].data for (t,W) in zip(ts,randn(length(ts)))]
    u
end #solve
