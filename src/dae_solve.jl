function dfdx(phi::ODEPhi, t::AbstractVector, θ, autodiff::Bool, differential_vars)
    if autodiff
        autodiff && throw(ArgumentError("autodiff not supported for DAE problem."))
    else
        dphi = (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
        # dim = size(dphi)[1]
        batch_size = size(t)[1]

        reduce(vcat,
            [if dv == true
                dphi[[i], :]
            else
                zeros(1, batch_size)
            end
             for (i, dv) in enumerate(differential_vars)])
    end
end

function inner_loss(phi::ODEPhi{C, T, U}, f, autodiff::Bool, t::AbstractVector, θ,
        p, differential_vars) where {C, T, U}
    out = Array(phi(t, θ))
    dphi = Array(dfdx(phi, t, θ, autodiff, differential_vars))
    dxdtguess = Array(dfdx(phi, t, θ, autodiff, differential_vars))
    arrt = Array(t)
    fs = reduce(hcat, [f(dphi[:, i], out[:, i], p, arrt[i]) for i in 1:size(out, 2)])
    sum(abs2, dxdtguess .- fs) / length(t)
end

function generate_loss(strategy::GridTraining, phi, f, autodiff::Bool, tspan, p, batch,
        differential_vars)
    ts = tspan[1]:(strategy.dx):tspan[2]
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    function loss(θ, _)
        sum(abs2, inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
    end
    return loss
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractDAEProblem,
        alg::NNODE,
        args...;
        dt = nothing,
        timeseries_errors = true,
        save_everystep = true,
        adaptive = false,
        abstol = 1.0f-6,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing,
        tstops = nothing)
    u0 = prob.u0
    du0 = prob.du0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain = alg.chain
    opt = alg.opt
    autodiff = alg.autodiff

    #train points generation
    init_params = alg.init_params

    # A logical array which declares which variables are thedifferential (non-algebraic) vars
    differential_vars = prob.differential_vars

    if chain isa Lux.AbstractExplicitLayer || chain isa Flux.Chain
        phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
    else
        error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
    end

    if isinplace(prob)
        throw(error("The NNODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    try
        phi(t0, init_params)
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end

    strategy = if alg.strategy === nothing
        if dt !== nothing
            GridTraining(dt)
        else
            QuadratureTraining(; quadrature_alg = QuadGKJL(),
                reltol = convert(eltype(u0), reltol),
                abstol = convert(eltype(u0), abstol), maxiters = maxiters,
                batch = 0)
        end
    else
        alg.strategy
    end

    batch = if alg.batch === nothing
        if strategy isa QuadratureTraining
            strategy.batch
        else
            true
        end
    else
        alg.batch
    end

    inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p, batch, differential_vars)

    # Creates OptimizationFunction Object from total_loss
    total_loss(θ, _) = inner_f(θ, phi)

    # Optimization Algo for Training Strategies
    opt_algo = Optimization.AutoZygote()
    # Creates OptimizationFunction Object from total_loss
    optf = OptimizationFunction(total_loss, opt_algo)

    iteration = 0
    callback = function (p, l)
        iteration += 1
        verbose && println("Current loss is: $l, Iteration: $iteration")
        l < abstol
    end
    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    #solutions at timepoints
    @show saveat
    if saveat isa Number
        ts = tspan[1]:saveat:tspan[2]
    elseif saveat isa AbstractArray
        ts = saveat
    elseif dt !== nothing
        ts = tspan[1]:dt:tspan[2]
    elseif save_everystep
        ts = range(tspan[1], tspan[2], length = 100)
    else
        ts = [tspan[1], tspan[2]]
    end

    if u0 isa Number
        u = [first(phi(t, res.u)) for t in ts]
    else
        u = [phi(t, res.u) for t in ts]
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, u;
        k = res, dense = true,
        calculate_error = false,
        retcode = ReturnCode.Success)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end #solve
