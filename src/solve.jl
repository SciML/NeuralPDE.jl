function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt = error("dt must be set."),
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

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
    phi(t) = u0 .+ (t .- tspan[1]).*chain(Tracker.collect([t]))

    #derivatives of a function f
    dfdx(t) = (phi(t+sqrt(eps(typeof(dt)))) - phi(t)) / sqrt(eps(typeof(dt)))
    #dfdx(t) = Flux.Tracker.forwarddiff(phi,t)
    #dfdx(t) = Tracker.collect([Flux.Tracker.gradient(t->phi(t)[i],t, nest=true) for i in 1:length(u0)])

    #loss function for training
    loss() = sum(abs2,sum(abs2,dfdx(t) - f(phi(t),p,t)) for t in ts)

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end
    Flux.train!(loss, ps, data, opt; cb = cb)

    #solutions at timepoints
    u = [phi(t).data for t in ts]

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
