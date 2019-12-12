using Flux.Tracker

struct NNODE{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
end
NNODE(chain;opt=Flux.ADAM(0.1)) = NNODE(chain,opt)

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt,
    initial_condition,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    u1 = initial_condition[1]
    u2 = 0.0f0

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
    phi(t) = u1 .+ (t .- tspan[1]).*chain(Tracker.collect([t]))
    if(length(initial_condition) == 2)
        u2 = initial_condition[2]
        phi(t) = (u1 .* (1 - (t .- tspan[1]))) .+ (u2 .* (t .- tspan[1])) .+ ((t .- tspan[1]) .* (1 - (t .- tspan[1])) .* chain(Tracker.collect([t])))
    end

    if u1 isa Number && length(initial_condition) != 2
        dfdx = t -> Tracker.gradient(t -> sum(phi(t)), t; nest = true)[1]
        loss = () -> sum(abs2,sum(abs2,dfdx(t) .- f(phi(t)[1],p,t)[1]) for t in ts)
    else
        dfdx = t -> (phi(t+sqrt(eps(typeof(dt)))) - phi(t)) / sqrt(eps(typeof(dt)))
        #dfdx(t) = Flux.Tracker.forwarddiff(phi,t)
        #dfdx(t) = Tracker.collect([Flux.Tracker.gradient(t->phi(t)[i],t, nest=true) for i in 1:length(u1)])
        #loss function for training
        loss = () -> sum(abs2,sum(abs2,dfdx(t) - f(phi(t),p,t)) for t in ts)
    end

    if length(initial_condition) == 2
        dfdx = t -> (phi(t+sqrt(eps(typeof(dt)))) - phi(t)) / sqrt(eps(typeof(dt)))
        df2dx = j -> Tracker.gradient(j -> sum(dfdx(j)), j; nest = true)[1]
        #df2dx = t -> Tracker.gradient(t -> sum(dfdx(t)), t; nest = true)[1]
        loss = () -> sum(abs2,sum(abs2,df2dx(t) .- f(phi(t)[1],p,t)[1]) for t in ts)
    end

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end
    Flux.train!(loss, ps, data, opt; cb = cb)

    #solutions at timepoints
    if u1 isa Number
        u = [phi(t)[1].data for t in ts]
    else
        u = [phi(t).data for t in ts]
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
