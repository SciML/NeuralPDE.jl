@concrete struct NNRODE <: NeuralPDEAlgorithm
    chain <: AbstractLuxLayer
    W
    opt
    init_params
    autodiff::Bool
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    kwargs
end

function NNRODE(chain, W, opt, init_params = nothing; strategy = nothing, autodiff = false,
        kwargs...)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return NNRODE(chain, W, opt, init_params, autodiff, strategy, kwargs)
end

@concrete struct RODEPhi
    u0
    t0
    smodel <: StatefulLuxLayer
end

RODEPhi(phi::ODEPhi) = RODEPhi(phi.u0, phi.t0, phi.smodel)

function (f::RODEPhi{<:Number})(t::Number, W, θ)
    return f.u0 + (t - f.t0) * first(f.smodel([t, W], θ.depvar))
end

function (f::RODEPhi{<:Number})(t::AbstractVector, W, θ)
    return f.u0 .+ (t' .- f.t0) .* f.smodel(vcat(t', W'), θ.depvar)
end

(f::RODEPhi)(t::Number, W, θ) = f.u0 .+ (t .- f.t0) .* f.smodel([t, W], θ.depvar)

function (f::RODEPhi)(t::AbstractVector, W, θ)
    return f.u0 .+ (t' .- f.t0) .* f.smodel(vcat(t', W'), θ.depvar)
end

function dfdx(phi::RODEPhi, t, θ, autodiff::Bool, W)
    autodiff && throw(ArgumentError("autodiff not supported for DAE problem."))
    ϵ = sqrt(eps(eltype(t)))
    return (phi(t .+ ϵ, W, θ) .- phi(t, W, θ)) ./ ϵ
end

function inner_loss(phi::RODEPhi, f, t::Number, θ, autodiff::Bool, p, W)
    return sum(abs2, dfdx(phi, t, θ, autodiff, W) .- f(phi(t, W, θ), p, t, W))
end

function inner_loss(phi::RODEPhi, f, t::AbstractVector, θ, autodiff::Bool, p, W)
    out = phi(t, W, θ)
    fs = reduce(hcat, [f(out[:, i], p, tᵢ, W[i]) for (i, tᵢ) in enumerate(t)])
    return sum(abs2, dfdx(phi, t, θ, autodiff, W) .- fs)
end

function generate_loss(strategy::GridTraining, phi::RODEPhi, f, autodiff::Bool, tspan, p, W)
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    ts = tspan[1]:(strategy.dx):tspan[2]
    return (θ, _) -> sum(abs2, inner_loss(phi, f, ts, θ, autodiff, p, W))
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractRODEProblem,
        alg::NNRODE,
        args...;
        dt,
        timeseries_errors = true,
        save_everystep = true,
        adaptive = false,
        abstol = 1.0f-6,
        verbose = false,
        maxiters = 100
)
    @assert !isinplace(prob) "The NNRODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t,W)."

    (; u0, tspan, f, p) = prob
    t0 = tspan[1]
    (; chain, opt, autodiff, init_params) = alg
    Wg = alg.W

    phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
    phi = RODEPhi(phi)
    init_params = ComponentArray(; depvar = init_params)

    strategy = if alg.strategy === nothing
        dt === nothing && error("`dt` is not defined")
        GridTraining(dt)
    end

    ts = if strategy isa GridTraining
        tspan[1]:(strategy.dx):tspan[2]
    else
        error("Only GridTraining is supported for now.")
    end

    Wprob = NoiseProblem(Wg, tspan)
    W = solve(Wprob; dt)

    inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p, W.W)
    total_loss(θ, _) = inner_f(θ, phi)
    optf = OptimizationFunction(total_loss, AutoZygote())

    plen = maxiters === nothing ? 6 : ndigits(maxiters)
    callback = function (p, l)
        if verbose
            if maxiters === nothing
                @printf("[NNRODE]\tIter: [%*d]\tLoss: %g\n", plen, p.iter, l)
            else
                @printf("[NNRODE]\tIter: [%*d/%d]\tLoss: %g\n", plen, p.iter, maxiters, l)
            end
        end
        return l < abstol
    end

    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    # solutions at timepoints
    noiseproblem = NoiseProblem(Wg, tspan)
    W = solve(noiseproblem; dt)
    if u0 isa Number
        u = [(phi(tᵢ, W.W[i], res.u)) for (i, tᵢ) in enumerate(ts)]
    else
        u = [(phi(tᵢ, W.W[i], res.u)) for (i, tᵢ) in enumerate(ts)]
    end

    sol = SciMLBase.build_solution(prob, alg, ts, u, W = W, calculate_error = false)
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    return sol
end
