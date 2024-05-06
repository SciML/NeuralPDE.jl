"""
    NNDAE(chain,
        OptimizationOptimisers.Adam(0.1),
        init_params = nothing;
        autodiff = false,
        kwargs...)

Algorithm for solving differential algebraic equationsusing a neural network. This is a specialization
of the physics-informed neural network which is used as a solver for a standard `DAEProblem`.

!!! warn

    Note that NNDAE only supports DAEs which are written in the out-of-place form, i.e.
    `du = f(du,u,p,t)`, and not `f(out,du,u,p,t)`. If not declared out-of-place, then the NNDAE
    will exit with an error.

## Positional Arguments

* `chain`: A neural network architecture, defined as either a `Flux.Chain` or a `Lux.AbstractExplicitLayer`.
* `opt`: The optimizer to train the neural network.
* `init_params`: The initial parameter of the neural network. By default, this is `nothing`
  which thus uses the random initialization provided by the neural network library.

## Keyword Arguments

* `autodiff`: The switch between automatic(not supported yet) and numerical differentiation for
              the PDE operators. The reverse mode of the loss function is always
              automatic differentiation (via Zygote), this is only for the derivative
              in the loss function (the derivative with respect to time).
* `strategy`: The training strategy used to choose the points for the evaluations.
              By default, `GridTraining` is used with `dt` if given.
"""
struct NNDAE{C, O, P, K, S <: Union{Nothing, AbstractTrainingStrategy}
} <: SciMLBase.AbstractDAEAlgorithm
    chain::C
    opt::O
    init_params::P
    autodiff::Bool
    strategy::S
    kwargs::K
end

function NNDAE(chain, opt, init_params = nothing; strategy = nothing, autodiff = false,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) &&
        (chain = adapt(FromFluxAdaptor(false, false), chain))
    NNDAE(chain, opt, init_params, autodiff, strategy, kwargs)
end

function dfdx(phi::ODEPhi, t::AbstractVector, θ, autodiff::Bool,
        differential_vars::AbstractVector)
    if autodiff
        autodiff && throw(ArgumentError("autodiff not supported for DAE problem."))
    else
        dphi = (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
        batch_size = size(t)[1]
        reduce(vcat,
            [dv ? dphi[[i], :] : zeros(1, batch_size)
             for (i, dv) in enumerate(differential_vars)])
    end
end

function inner_loss(phi::ODEPhi{C, T, U}, f, autodiff::Bool, t::AbstractVector, θ,
        p, differential_vars::AbstractVector) where {C, T, U}
    out = Array(phi(t, θ))
    dphi = Array(dfdx(phi, t, θ, autodiff, differential_vars))
    arrt = Array(t)
    loss = reduce(hcat, [f(dphi[:, i], out[:, i], p, arrt[i]) for i in 1:size(out, 2)])
    sum(abs2, loss) / length(t)
end

function generate_loss(strategy::GridTraining, phi, f, autodiff::Bool, tspan, p,
        differential_vars::AbstractVector)
    ts = tspan[1]:(strategy.dx):tspan[2]
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    function loss(θ, _)
        sum(abs2, inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
    end
    return loss
end

function generate_loss(
        strategy::WeightedIntervalTraining, phi, f, autodiff::Bool, tspan, p,
        differential_vars::AbstractVector)
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    minT = tspan[1]
    maxT = tspan[2]

    weights = strategy.weights ./ sum(strategy.weights)

    N = length(weights)
    points = strategy.points

    difference = (maxT - minT) / N

    data = Float64[]
    for (index, item) in enumerate(weights)
        temp_data = rand(1, trunc(Int, points * item)) .* difference .+ minT .+
                    ((index - 1) * difference)
        data = append!(data, temp_data)
    end

    ts = data

    function loss(θ, _)
        sum(inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
    end
    return loss
end

function SciMLBase.__solve(prob::SciMLBase.AbstractDAEProblem,
        alg::NNDAE,
        args...;
        dt = nothing,
        # timeseries_errors = true,
        save_everystep = true,
        # adaptive = false,
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

    # A logical array which declares which variables are the differential (non-algebraic) vars
    differential_vars = prob.differential_vars

    if chain isa Lux.AbstractExplicitLayer || chain isa Flux.Chain
        phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
        init_params = ComponentArrays.ComponentArray(;
            depvar = ComponentArrays.ComponentArray(init_params))
    else
        error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
    end

    if isinplace(prob)
        throw(error("The NNODE solver only supports out-of-place DAE definitions, i.e. du=f(u,p,t)."))
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
            error("dt is not defined")
        end
    else
        alg.strategy
    end

    inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p, differential_vars)

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

    sol = SciMLBase.build_solution(prob, alg, ts, u;
        k = res, dense = true,
        calculate_error = false,
        retcode = ReturnCode.Success,
        original = res,
        resid = res.objective)
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
