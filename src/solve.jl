function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt = error("dt must be set."),
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1e-6,
    maxiters = 100)

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    ps     = Flux.Params(chain)
    data   = Iterators.repeated((), maxiters)

    #train points generation
    ts = tspan[1]:dt:tspan[2]

    #The phi trial solution
    phi(t) = u0 .+ x.*chain([t])

    #derivatives of a function f
    dfdx(t) = Tracker.gradient(t -> chain([t]), t; nest = true)

    #loss function for training
    loss() = sum(abs2, dfdx(t) - f(phi(t),p,t) for t in ts)

    @time for iters=1:maxiters
        Flux.train!(loss, ps, data, opt)
        if mod(iters,50) == 0
            loss_value = loss().data
            println((:iteration,iters,:loss,loss_value))
            if loss_value < abstol
                break
            end
        end
    end

    #solutions at timepoints
    u = [phi(t)[1].data for i in t]

    sol = DiffEqBase.build_solution(prob,alg,x,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
