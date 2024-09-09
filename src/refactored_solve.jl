abstract type NeuralPDEAlgorithm <: SciMLBase.AbstractODEAlgorithm end

struct NNODE{C, O, P, B, PE, K, AL <: Union{Nothing, Function},
    S <: Union{Nothing, AbstractTrainingStrategy}} <:NeuralPDEAlgorithm
    chain::C
    opt::O
    init_params::P
    autodiff::Bool
    batch::B
    strategy::S
    param_estim::PE
    additional_loss::AL
    kwargs::K
end

function NNODE(chain, opt, init_params = nothing;
        strategy = nothing,
        autodiff = false, batch = true, param_estim = false, additional_loss = nothing, kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) &&
        (chain = adapt(FromFluxAdaptor(false, false), chain))
    NNODE(chain, opt, init_params, autodiff, batch,
        strategy, param_estim, additional_loss, kwargs)
end

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

mutable struct ODEPhi{C, T, U, S}
    chain::C
    t0::T
    u0::U
    st::S
    function ODEPhi(chain::Lux.AbstractExplicitLayer, t::Number, u0, st)
        new{typeof(chain), typeof(t), typeof(u0), typeof(st)}(chain, t, u0, st)
    end
end

function generate_phi_θ(chain::Lux.AbstractExplicitLayer, t, u0, init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    isnothing(init_params) && (init_params = θ)
    ODEPhi(chain, t, u0, st), init_params
end

function (f::ODEPhi{C, T, U})(t::Number,
        θ) where {C <: Lux.AbstractExplicitLayer, T, U <: Number}
    y, st = f.chain(
        adapt(parameterless_type(ComponentArrays.getdata(θ.depvar)), [t]), θ.depvar, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.u0 + (t - f.t0) * first(y)
end

function (f::ODEPhi{C, T, U})(t::AbstractVector,
        θ) where {C <: Lux.AbstractExplicitLayer, T, U <: Number}
    # Batch via data as row vectors
    y, st = f.chain(
        adapt(parameterless_type(ComponentArrays.getdata(θ.depvar)), t'), θ.depvar, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.u0 .+ (t' .- f.t0) .* y
end

function (f::ODEPhi{C, T, U})(t::Number, θ) where {C <: Lux.AbstractExplicitLayer, T, U}
    y, st = f.chain(
        adapt(parameterless_type(ComponentArrays.getdata(θ.depvar)), [t]), θ.depvar, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.u0 .+ (t .- f.t0) .* y
end

function (f::ODEPhi{C, T, U})(t::AbstractVector,
        θ) where {C <: Lux.AbstractExplicitLayer, T, U}
    # Batch via data as row vectors
    y, st = f.chain(
        adapt(parameterless_type(ComponentArrays.getdata(θ.depvar)), t'), θ.depvar, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    f.u0 .+ (t' .- f.t0) .* y
end

function dfdx end

function dfdx(phi::ODEPhi{C, T, U}, t::Number, θ,
        autodiff::Bool) where {C, T, U <: Number}
    if autodiff
        ForwardDiff.derivative(t -> phi(t, θ), t)
    else
        (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, θ)) / sqrt(eps(typeof(t)))
    end
end

function dfdx(phi::ODEPhi{C, T, U}, t::Number, θ,
        autodiff::Bool) where {C, T, U <: AbstractVector}
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, θ), t)
    else
        (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, θ)) / sqrt(eps(typeof(t)))
    end
end

function dfdx(phi::ODEPhi, t::AbstractVector, θ, autodiff::Bool; differential_vars::AbstractVector) # SHOULD IT BE OPTIONAL VS KEYWORD ARGUMENT
    if differential_vars isa Nothing # DOES THIS TEST CONDITION MAKE SENSE. ALSO CHECK IF ELSE IS SYNTICALLY CORRECT
        if autodiff
            ForwardDiff.jacobian(t -> phi(t, θ), t)
        else
            (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
        end
    else
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
end

function inner_loss end

function inner_loss(phi::ODEPhi{C, T, U}, f, autodiff::Bool, t::Number, θ,
        p, param_estim::Bool) where {C, T, U <: Number}
    p_ = param_estim ? θ.p : p
    sum(abs2, ode_dfdx(phi, t, θ, autodiff) - f(phi(t, θ), p_, t))
end

function inner_loss(phi::ODEPhi{C, T, U}, f, autodiff::Bool, t::AbstractVector, θ,
        p, param_estim::Bool) where {C, T, U <: Number}
    p_ = param_estim ? θ.p : p
    out = phi(t, θ)
    fs = reduce(hcat, [f(out[i], p_, t[i]) for i in axes(out, 2)])
    dxdtguess = Array(ode_dfdx(phi, t, θ, autodiff))
    sum(abs2, dxdtguess .- fs) / length(t)
end

function inner_loss(phi::ODEPhi{C, T, U}, f, autodiff::Bool, t::Number, θ,
    p; param_estim::Bool, differential_vars::AbstractVector) where {C, T, U} # SHOULD IT BE OPTIONAL VS KEYWORD ARGUMENT
    if !(param_estim isa Nothing)
        p_ = param_estim ? θ.p : p
        sum(abs2, ode_dfdx(phi, t, θ, autodiff) .- f(phi(t, θ), p_, t))
    end
    if !(differential_vars isa Nothing)
        dphi = dfdx(phi, t, θ, autodiff,differential_vars)
        sum(abs2, f(dphi, phi(t, θ), p, t))
    end
end

function inner_loss(phi::ODEPhi{C, T, U}, f, autodiff::Bool, t::AbstractVector, θ,
    p; param_estim::Bool, differential_vars::AbstractVector) where {C, T, U}
    if !(param_estim isa Nothing)
        p_ = param_estim ? θ.p : p
        out = Array(phi(t, θ))
        arrt = Array(t)
        fs = reduce(hcat, [f(out[:, i], p_, arrt[i]) for i in 1:size(out, 2)])
        dxdtguess = Array(ode_dfdx(phi, t, θ, autodiff))
        sum(abs2, dxdtguess .- fs) / length(t)
    end
    if !(differential_vars isa Nothing)
        out = Array(phi(t, θ))
        dphi = Array(dfdx(phi, t, θ, autodiff, differential_vars))
        arrt = Array(t)
        loss = reduce(hcat, [f(dphi[:, i], out[:, i], p, arrt[i]) for i in 1:size(out, 2)])
        sum(abs2, loss) / length(t)
    end
end

function generate_loss(strategy::QuadratureTraining, phi, f, autodiff::Bool, tspan, p;
    batch, param_estim::Bool, differential_vars::AbstractVector)

    if !(param_estim isa Nothing)
        integrand(t::Number, θ) = abs2(inner_loss(phi, f, autodiff, t, θ, p, param_estim))
    end

    if !(differential_vars isa Nothing)
        integrand(t::Number, θ) = abs2(inner_loss(phi, f, autodiff, t, θ, p, differential_vars))
    end

    function integrand(ts, θ)
        if !(param_estim isa Nothing)
            [abs2(inner_loss(phi, f, autodiff, t, θ, p, param_estim)) for t in ts]
        end
        if !(differential_vars isa Nothing)
            [sum(abs2, inner_loss(phi, f, autodiff, t, θ, p, differential_vars)) for t in ts]
        end
    end

    function loss(θ, _)
        intf = BatchIntegralFunction(integrand, max_batch = strategy.batch)
        intprob = IntegralProblem(intf, (tspan[1], tspan[2]), θ)
        sol = solve(intprob, strategy.quadrature_alg; abstol = strategy.abstol,
            reltol = strategy.reltol, maxiters = strategy.maxiters)
        sol.u
    end
    return loss
end

function generate_loss(strategy::GridTraining, phi, f, autodiff::Bool, tspan, p;
    batch, param_estim::Bool, differential_vars::AbstractVector)
    ts = tspan[1]:(strategy.dx):tspan[2]
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    function loss(θ, _)
        if !(param_estim isa Nothing)
            if batch
                inner_loss(phi, f, autodiff, ts, θ, p, param_estim)
            else
                sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in ts])
            end
        end
        if !(differential_vars isa Nothing)
            sum(abs2, inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
        end
    end
    return loss
end

function generate_loss(strategy::StochasticTraining, phi, f, autodiff::Bool, tspan, p;
    batch, param_estim::Bool, differential_vars::AbstractVector)
    autodiff && throw(ArgumentError("autodiff not supported for StochasticTraining."))
    function loss(θ, _)
        if !(param_estim isa Nothing)
            ts = adapt(parameterless_type(θ),
                [(tspan[2] - tspan[1]) * rand() + tspan[1] for i in 1:(strategy.points)])
            if batch
                inner_loss(phi, f, autodiff, ts, θ, p, param_estim)
            else
                sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in ts])
            end
        end
        if !(differential_vars isa Nothing)
            ts = adapt(parameterless_type(θ),
            [(tspan[2] - tspan[1]) * rand() + tspan[1] for i in 1:(strategy.points)])
            sum(inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
        end
    end
    return loss
end

function generate_loss(strategy::WeightedIntervalTraining, phi, f, autodiff::Bool, tspan, p,
    batch, param_estim::Bool)

    autodiff && throw(ArgumentError("autodiff not supported for WeightedIntervalTraining."))
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
        if !(param_estim isa Nothing)
            if batch
                inner_loss(phi, f, autodiff, ts, θ, p, param_estim)
            else
                sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in ts])
            end
        end
        if !(differential_vars isa Nothing)
            sum(inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
        end
    end
    return loss
end

function evaluate_tstops_loss(phi, f, autodiff::Bool, tstops, p, batch, param_estim::Bool)
    function loss(θ, _)
        if batch
            inner_loss(phi, f, autodiff, tstops, θ, p, param_estim)
        else
            sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in tstops])
        end
    end
    return loss
end

function generate_loss(strategy::QuasiRandomTraining, phi, f, autodiff::Bool, tspan)
    error("QuasiRandomTraining is not supported by NNODE since it's for high dimensional spaces only. Use StochasticTraining instead.")
end

struct NNODEInterpolation{T <: ODEPhi, T2}
    phi::T
    θ::T2
end
(f::NNODEInterpolation)(t, idxs::Nothing, ::Type{Val{0}}, p, continuity) = f.phi(t, f.θ)
(f::NNODEInterpolation)(t, idxs, ::Type{Val{0}}, p, continuity) = f.phi(t, f.θ)[idxs]

function (f::NNODEInterpolation)(t::Vector, idxs::Nothing, ::Type{Val{0}}, p, continuity)
    out = f.phi(t, f.θ)
    SciMLBase.RecursiveArrayTools.DiffEqArray([out[:, i] for i in axes(out, 2)], t)
end

function (f::NNODEInterpolation)(t::Vector, idxs, ::Type{Val{0}}, p, continuity)
    out = f.phi(t, f.θ)
    SciMLBase.RecursiveArrayTools.DiffEqArray([out[idxs, i] for i in axes(out, 2)], t)
end

SciMLBase.interp_summary(::NNODEInterpolation) = "Trained neural network interpolation"
SciMLBase.allowscomplex(::NNODE) = true

function SciMLBase.__solve(prob,
    alg,
    args...;
    dt = nothing,
    save_everystep = true,
    abstol = 1.0f-6,
    reltol = 1.0f-3,
    verbose = false,
    saveat = nothing,
    maxiters = nothing,
    tstops = nothing)

    if prob::SciMLBase.AbstractODEProblem && alg::NNODE
        timeseries_errors = true
        adaptive = false
    end
    u0 = prob.u0

    if prob::SciMLBase.AbstractDAEProblem && alg::NNDAE
        du0 = prob.du0
    end

    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    if prob::SciMLBase.AbstractODEProblem && alg::NNODE
        param_estim = alg.param_estim
    end

    #hidden layer
    chain = alg.chain
    opt = alg.opt
    autodiff = alg.autodiff

    #train points generation
    init_params = alg.init_params

    if prob::SciMLBase.AbstractODEProblem && alg::NNODE
        !(chain isa Lux.AbstractExplicitLayer) && 
            error("Only Lux.AbstractExplicitLayer neural networks are supported")
        phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
        (recursive_eltype(init_params) <: Complex &&
        alg.strategy isa QuadratureTraining) &&
            error("QuadratureTraining cannot be used with complex parameters. Use other strategies.")

        init_params = if alg.param_estim
            ComponentArrays.ComponentArray(;
                depvar = ComponentArrays.ComponentArray(init_params), p = prob.p)
        else
            ComponentArrays.ComponentArray(;
                depvar = ComponentArrays.ComponentArray(init_params))
        end
    end

    if prob::SciMLBase.AbstractDAEProblem && alg::NNDAE
        # A logical array which declares which variables are the differential (non-algebraic) vars
        differential_vars = prob.differential_vars
        if chain isa Lux.AbstractExplicitLayer || chain isa Flux.Chain
            phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
            init_params = ComponentArrays.ComponentArray(;
                depvar = ComponentArrays.ComponentArray(init_params))
        else
            error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
        end
    end

    isinplace(prob) &&
        throw(error("The NNODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

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

    if prob::SciMLBase.AbstractODEProblem && alg::NNODE
        batch = alg.batch
        inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p; batch, param_estim)
        additional_loss = alg.additional_loss
        (param_estim && isnothing(additional_loss)) &&
            throw(ArgumentError("Please provide `additional_loss` in `NNODE` for parameter estimation (`param_estim` is true)."))

        # Creates OptimizationFunction Object from total_loss
        function total_loss(θ, _)
            L2_loss = inner_f(θ, phi)
            if !(additional_loss isa Nothing)
                L2_loss = L2_loss + additional_loss(phi, θ)
            end
            if !(tstops isa Nothing)
                num_tstops_points = length(tstops)
                tstops_loss_func = evaluate_tstops_loss(
                    phi, f, autodiff, tstops, p, batch, param_estim)
                tstops_loss = tstops_loss_func(θ, phi)
                if strategy isa GridTraining
                    num_original_points = length(tspan[1]:(strategy.dx):tspan[2])
                elseif strategy isa Union{WeightedIntervalTraining, StochasticTraining}
                    num_original_points = strategy.points
                else
                    return L2_loss + tstops_loss
                end
                total_original_loss = L2_loss * num_original_points
                total_tstops_loss = tstops_loss * num_tstops_points
                total_points = num_original_points + num_tstops_points
                L2_loss = (total_original_loss + total_tstops_loss) / total_points
            end
            return L2_loss
        end

        # Choice of Optimization Algo for Training Strategies
        opt_algo = if strategy isa QuadratureTraining
            Optimization.AutoForwardDiff()
        else
            Optimization.AutoZygote()
        end
        # Creates OptimizationFunction Object from total_loss
    end

    if prob::SciMLBase.AbstractDAEProblem && alg::NNDAE
        inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p; differential_vars)
        # Creates OptimizationFunction Object from total_loss
        total_loss(θ, _) = inner_f(θ, phi)

        # Optimization Algo for Training Strategies
        opt_algo = Optimization.AutoZygote()
        # Creates OptimizationFunction Object from total_loss
    end

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

    if prob::SciMLBase.AbstractODEProblem && alg::NNODE
        sol = SciMLBase.build_solution(prob, alg, ts, u;
        k = res, dense = true,
        interp = NNODEInterpolation(phi, res.u),
        calculate_error = false,
        retcode = ReturnCode.Success,
        original = res,
        resid = res.objective)
    end

    if prob::SciMLBase.AbstractDAEProblem && alg::NNDAE
        sol = SciMLBase.build_solution(prob, alg, ts, u;
        k = res, dense = true,
        calculate_error = false,
        retcode = ReturnCode.Success,
        original = res,
        resid = res.objective)
    end

    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end





