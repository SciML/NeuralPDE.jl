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
    uElType = eltype(u0)
    tType = typeof(tspan[1])
    outdim = length(u0)

    #hidden layer
    hl_width = alg.hl_width

    #train points generation
    dtrn = generate_data(tspan[1],tspan[2],dt,atype=tType)
    #The phi trial solution
    phi(P,t) = (u0 .+ t*predict(P,t))[1]

    #initialization of weights and bias
    w = init_params(uElType,hl_width)
    #initialization of optimization parameters (Adam by default for now)
    lr_ = 0.1
    beta1_ = 0.9
    beta2_ = 0.95
    eps_ = 1e-6
    prms = Any[]

    for i=1:length(w)
    prm = Adam(lr=lr_, beta1=beta1_, beta2=beta2_, eps=eps_)
    push!(prms, prm)
    end

    @time for iters=1:maxiters
            train(w, prms, dtrn, f, p,phi)
            loss = test(w,dtrn,f,p,phi)
            if mod(iters,100) == 0
                println((:iteration,iters,:loss,loss))
            end

            if loss < 10^(-8.0)
                break
            end
        end

    #solutions at timepoints
    u = [phi(w,x) for x in dtrn]

    sol = DiffEqBase.build_solution(prob,alg,dtrn,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
