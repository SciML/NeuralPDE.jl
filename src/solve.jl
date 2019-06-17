function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt = error("dt must be set."),
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    maxiters = 100)

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #types and dimensions
    # uElType = eltype(u0)
    # tType = typeof(tspan[1])
    # outdim = length(u0)

    #hidden layer
    hl_width = alg.hl_width

    #initialization of weights and bias
    P = init_params(hl_width)

    #The phi trial solution
    phi(P,x) = u0 .+ x.*predict(P,x)

    #train points generation
    x = generate_data(tspan[1],tspan[2],dt)
    y = [f(phi(P, i)[1].data, p, i) for i in x]
    px =Flux.param(x)
    data = [(px, y)]

    #initialization of optimization parameters (ADAM by default for now)
    η = 0.1
    β1 = 0.9
    β2 = 0.95
    opt = Flux.ADAM(η, (β1, β2))

    ps = Flux.Params(P)

    #derivatives of a function f
    dfdx(i) = Tracker.gradient(() -> sum(phi(P,i)), Flux.params(i); nest = true)
    #loss function for training
    loss(x, y) = sum(abs2, [dfdx(i)[i] for i in x] .- y)

    @time for iters=1:maxiters
        Flux.train!(loss, ps, data, opt)
        if mod(iters,50) == 0
            loss_value = loss(px,y).data
            println((:iteration,iters,:loss,loss_value))
            if loss_value < 10^(-6.0)
                break
            end
        end
    end

    #solutions at timepoints
    u = [phi(P,i)[1].data for i in x]

    sol = DiffEqBase.build_solution(prob,alg,x,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
