@concrete struct NNSDE
    chain <: AbstractLuxLayer
    opt
    init_params
    autodiff::Bool
    batch
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    param_estim
    additional_loss <: Union{Nothing, Function}
    kwargs
end

function NNSDE(chain, opt, init_params = nothing; strategy = nothing, autodiff = false,
        batch = true, param_estim = false, additional_loss = nothing, kwargs...)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return NNSDE(chain, opt, init_params, autodiff, batch,
        strategy, param_estim, additional_loss, kwargs)
end

"""
    SDEPhi(chain::Lux.AbstractLuxLayer, t, u0, st)

Internal struct, used for representing the SDE solution as a neural network in a form that
respects boundary conditions, i.e. `phi(inp) = u0 + inp[1]*NN(inp)`.
"""
@concrete struct SDEPhi
    u0
    t0
    smodel <: StatefulLuxLayer
end

function SDEPhi(model::AbstractLuxLayer, t0::Number, u0, st)
    return SDEPhi(u0, t0, StatefulLuxLayer{true}(model, nothing, st))
end

function (f::SDEPhi)(inp, θ)
    dev = safe_get_device(θ)
    return f(dev, safe_expand(dev, inp), θ)
end

function (f::SDEPhi{<:Number})(dev, inp::Array{<:Number, 1}, θ)
    res = only(cdev(f.smodel(dev(inp), θ.depvar)))
    return f.u0 + (inp[1] - f.t0) * res
end

function (f::SDEPhi{<:Number})(_, inp::AbstractVector{<:Array{<:Number, 1}}, θ)
    return [f.u0 .+ (inpi[1] .- f.t0) .* f.smodel(inpi', θ.depvar) for inpi in inp]
end

function (f::SDEPhi)(dev, inp::Array{<:Number, 1}, θ)
    return dev(f.u0) .+ (inp[1] - f.t0) .* f.smodel(dev(inp), θ.depvar)
end

function (f::SDEPhi)(dev, inp::AbstractVector{<:Array{<:Number, 1}}, θ)
    return [dev(f.u0) .+ (inpi[1] - f.t0) .* f.smodel(inpi, θ.depvar) for inpi in inp]
end

function generate_phi(chain::AbstractLuxLayer, t, u0, ::Nothing)
    θ, st = LuxCore.setup(Random.default_rng(), chain)
    return SDEPhi(chain, t, u0, st), θ
end

function generate_phi(chain::AbstractLuxLayer, t, u0, init_params)
    st = LuxCore.initialstates(Random.default_rng(), chain)
    return SDEPhi(chain, t, u0, st), init_params
end

"""
    ∂u_∂t(phi, inp, θ, autodiff)

Computes for sde's solution u, u' using either forward-mode automatic differentiation or numerical differentiation.
"""
function ∂u_∂t end

# earlier input as number or abstract vector, now becomes array1 or matrix
# inp first col must be time, rest all z_i's
function ∂u_∂t(phi::SDEPhi, inp::Array{<:Number, 1}, θ, autodiff::Bool)
    autodiff && return ForwardDiff.gradient(t -> (phi(vcat(t, inp[2:end]), θ)), inp[1])
    ϵ = sqrt(eps(eltype(inp[1])))
    return (phi(vcat(inp[1] + ϵ, inp[2:end]), θ) .- phi(inp, θ)) ./ ϵ
end

"""
    inner_sde_loss(phi, f, autodiff, t, θ, p, param_estim)

Simple L2 inner loss for the SDE at a time `t` and random variables z_i with parameters `θ` of the neural network.
"""
function inner_sde_loss end

function inner_sde_loss(
        phi::SDEPhi, f, g, autodiff::Bool, inp::Array{<:Number, 1}, θ, p, param_estim::Bool)
    p_ = param_estim ? θ.p : p
    u = phi(inp, θ)
    fs = if phi.u0 isa Number
        f(u, p_, inp[1]) +
        g(u, p_, inp[1]) * 2^(1 / 2) *
        sum(inp[1 + j] * cos((j - 1 / 2)pi * inp[1]) for j in 1:(length(inp) - 1))
    else
        # will be vector in multioutput case
        [f(u[:, 1], p_, inp[1]) +
         g(u[:, 1], p_, inp[1]) * 2^(1 / 2) *
         sum(inp[1 + j] * cos((j - 1 / 2)pi * inp[1]) for j in 1:(length(inp) - 1))]
    end
    dudt = ∂u_∂t(phi, inp, θ, autodiff)
    return sum(abs2, fs .- dudt)
end

# NNODE, NNSDE take only a single NN, multioutput or not
# instead of matrix Nx(t+n_z) dim array
function inner_sde_loss(
        phi::SDEPhi, f, g, autodiff::Bool, inp::Array{<:Array{<:Number, 1}}, θ, p, param_estim::Bool)
    p_ = param_estim ? θ.p : p
    u = [phi(inpi, θ) for inpi in inp]
    fs = if phi.u0 isa Number
        reduce(hcat,
            [f(u[i][1], p_, inpi[1]) +
             g(u[i][1], p_, inpi[1]) * 2^(1 / 2) *
             sum(inpi[1 + j] * cos((j - 1 / 2)pi * inpi[1]) for j in 1:(length(inpi) - 1))
             for (i, inpi) in enumerate(inp)])
    else
        reduce(hcat,
            [f(u[i][:, 1], p_, inpi[1]) +
             g(u[i][:, 1], p_, inpi[1]) * 2^(1 / 2) *
             sum(inpi[1 + j] * cos((j - 1 / 2)pi * inpi[1]) for j in 1:(length(inpi) - 1))
             for (i, inpi) in enumerate(inp)])
    end
    dudt = [∂u_∂t(phi, inpi, θ, autodiff) for inpi in inp]
    return sum(abs2, fs .- dudt) / length(inp)
end

"""
    generate_loss(strategy, phi, f, autodiff, tspan, p, batch, param_estim)

Representation of the loss function, parametric on the training strategy `strategy`.
"""
function generate_loss(
        strategy::QuadratureTraining, phi, f, g, autodiff::Bool, tspan, n_z, p,
        batch, param_estim::Bool)
    function integrand(t::Number, θ)
        abs2(inner_sde_loss(
            phi, f, g, autodiff,
            vcat(t, rand(Distributions.truncated(Normal(0, 1), -1, 1), n_z)),
            θ, p, param_estim))
    end

    function integrand(ts, θ)
        return [abs2(inner_sde_loss(
                    phi, f, g, autodiff,
                    vcat(t, rand(Distributions.truncated(Normal(0, 1), -1, 1), n_z)),
                    θ, p, param_estim))
                for t in ts]
    end

    function loss(θ, _)
        intf = BatchIntegralFunction(integrand, max_batch = strategy.batch)
        intprob = IntegralProblem(intf, (tspan[1], tspan[2]), θ)
        sol = solve(intprob, strategy.quadrature_alg; strategy.abstol,
            strategy.reltol, strategy.maxiters)
        return sol.u
    end

    return loss
end

function add_rand_coeff(times, n_z)
    return [vcat(time, rand(Distributions.truncated(Normal(0, 1), -1, 1), n_z))
            for time in times]
end

function generate_loss(
        strategy::GridTraining, phi, f, g, autodiff::Bool, tspan, n_z, p, batch, param_estim::Bool)
    ts = tspan[1]:(strategy.dx):tspan[2]
    inp = add_rand_coeff(collect(ts), n_z)
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    batch &&
        return (θ, _) -> inner_sde_loss(
            phi, f, g, autodiff, inp, θ, p, param_estim)
    return (θ, _) -> sum([inner_sde_loss(phi, f, g, autodiff, input, θ, p, param_estim)
                          for input in inp])
end

function generate_loss(
        strategy::StochasticTraining, phi, f, g, autodiff::Bool, tspan, n_z, p,
        batch, param_estim::Bool)
    autodiff && throw(ArgumentError("autodiff not supported for StochasticTraining."))

    return (θ, _) -> begin
        T = promote_type(eltype(tspan[1]), eltype(tspan[2]))
        ts = ((tspan[2] - tspan[1]) .* rand(T, strategy.points) .+ tspan[1])
        inp = add_rand_coeff(ts, n_z)

        if batch
            inner_sde_loss(phi, f, g, autodiff, inp, θ, p, param_estim)
        else
            sum([inner_sde_loss(phi, f, g, autodiff, input, θ, p, param_estim)
                 for input in inp])
        end
    end
end

function generate_loss(
        strategy::WeightedIntervalTraining, phi, f, g, autodiff::Bool, tspan, n_z, p,
        batch, param_estim::Bool)
    autodiff && throw(ArgumentError("autodiff not supported for WeightedIntervalTraining."))
    minT, maxT = tspan
    weights = strategy.weights ./ sum(strategy.weights)
    N = length(weights)
    difference = (maxT - minT) / N

    ts = eltype(difference)[]
    for (index, item) in enumerate(weights)
        temp_data = rand(1, trunc(Int, strategy.points * item)) .* difference .+ minT .+
                    ((index - 1) * difference)
        append!(ts, temp_data)
    end
    inp = add_rand_coeff(ts, n_z)

    batch &&
        return (θ, _) -> inner_sde_loss(
            phi, f, g, autodiff, inp, θ, p, param_estim)
    return (θ, _) -> sum([inner_sde_loss(phi, f, g, autodiff, input, θ, p, param_estim)
                          for input in inp])
end

function evaluate_tstops_loss(
        phi, f, g, autodiff::Bool, tstops, n_z, p, batch, param_estim::Bool)
    inp = add_rand_coeff(tstops, n_z)
    batch &&
        return (θ, _) -> inner_sde_loss(
            phi, f, g, autodiff, inp, θ, p, param_estim)
    return (θ, _) -> sum([inner_sde_loss(phi, f, g, autodiff, t, θ, p, param_estim)
                          for t in inp])
end

function generate_loss(::QuasiRandomTraining, phi, f, g, autodiff::Bool,
        tspan, n_z, p, batch, param_estim::Bool)
    error("QuasiRandomTraining is not supported by NNODE since it's for high dimensional \
           spaces only. Use StochasticTraining instead.")
end

@concrete struct NNSDEInterpolation
    phi <: SDEPhi
    θ
end

(f::NNSDEInterpolation)(inp, ::Nothing, ::Type{Val{0}}, p, continuity) = f.phi(inp, f.θ)
(f::NNSDEInterpolation)(inp, idxs, ::Type{Val{0}}, p, continuity) = f.phi(inp, f.θ)[idxs]

function (f::NNSDEInterpolation)(
        inp::Array{<:Number, 1}, ::Nothing, ::Type{Val{0}}, p, continuity)
    out = f.phi(inp, f.θ)
    return DiffEqArray([out[:, i] for i in axes(out, 2)], inp)
end

function (f::NNSDEInterpolation)(
        inp::Array{<:Number, 1}, idxs, ::Type{Val{0}}, p, continuity)
    out = f.phi(inp, f.θ)
    return DiffEqArray([out[idxs, i] for i in axes(out, 2)], inp)
end

SciMLBase.interp_summary(::NNSDEInterpolation) = "Trained neural network interpolation"
SciMLBase.allowscomplex(::NNSDE) = true

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSDEProblem,
        alg::NNSDE,
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
        tstops = nothing
)
    (; u0, tspan, f, g, p) = prob
    # rescaling timespan so KKL expansion can be applied for loss formulation
    tspan = tspan ./ tspan[end]
    t0 = tspan[1]
    (; param_estim, chain, opt, autodiff, init_params, batch, additional_loss) = alg
    n_z = chain[1].in_dims - 1

    phi, init_params = generate_phi(chain, t0, u0, init_params)

    (recursive_eltype(init_params) <: Complex && alg.strategy isa QuadratureTraining) &&
        error("QuadratureTraining cannot be used with complex parameters. Use other strategies.")

    init_params = if alg.param_estim
        ComponentArray(; depvar = init_params, p)
    else
        ComponentArray(; depvar = init_params)
    end

    @assert !isinplace(prob) "The NNSDE solver only supports out-of-place SDE definitions, i.e. du=f(u,p,t) + g(u,p,t)*dW(t)"

    strategy = if alg.strategy === nothing
        if dt !== nothing
            GridTraining(dt)
        else
            QuadratureTraining(; quadrature_alg = QuadGKJL(),
                reltol = convert(eltype(u0), reltol), abstol = convert(eltype(u0), abstol),
                maxiters, batch = 0)
        end
    else
        alg.strategy
    end

    inner_f = generate_loss(
        strategy, phi, f, g, autodiff, tspan, n_z, p, batch, param_estim)

    (param_estim && additional_loss === nothing) &&
        throw(ArgumentError("Please provide `additional_loss` in `NNSDE` for parameter estimation (`param_estim` is true)."))

    # Creates OptimizationFunction Object from total_loss
    function total_loss(θ, _)
        L2_loss = inner_f(θ, phi)
        if additional_loss !== nothing
            L2_loss = L2_loss + additional_loss(phi, θ)
        end
        if tstops !== nothing
            num_tstops_points = length(tstops)
            tstops_loss_func = evaluate_tstops_loss(
                phi, f, g, autodiff, tstops, n_z, p, batch, param_estim)
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

    opt_algo = ifelse(strategy isa QuadratureTraining, AutoForwardDiff(), AutoZygote())
    optf = OptimizationFunction(total_loss, opt_algo)

    plen = maxiters === nothing ? 6 : ndigits(maxiters)
    callback = function (p, l)
        if verbose
            if maxiters === nothing
                @printf("[NNSDE]\tIter: [%*d]\tLoss: %g\n", plen, p.iter, l)
            else
                @printf("[NNSDE]\tIter: [%*d/%d]\tLoss: %g\n", plen, p.iter, maxiters, l)
            end
        end
        return l < abstol
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
    inputs = add_rand_coeff(ts, n_z)

    if u0 isa Number
        u = [first(phi(input, res.u)) for input in inputs]
    else
        u = [phi(input, res.u) for input in inputs]
    end

    sol = SciMLBase.build_solution(prob, alg, inputs, u; k = res, dense = true,
        interp = NNSDEInterpolation(phi, res.u), calculate_error = false,
        retcode = ReturnCode.Success, original = res, resid = res.objective)

    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(
            sol; timeseries_errors = true, dense_errors = false)

    return sol
end