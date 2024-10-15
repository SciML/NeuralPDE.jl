"""
    NNDAE(chain, opt, init_params = nothing; autodiff = false, kwargs...)

Algorithm for solving differential algebraic equationsusing a neural network. This is a
specialization of the physics-informed neural network which is used as a solver for a
standard `DAEProblem`.

!!! warning

    Note that NNDAE only supports DAEs which are written in the out-of-place form, i.e.
    `du = f(du,u,p,t)`, and not `f(out,du,u,p,t)`. If not declared out-of-place, then the
    NNDAE will exit with an error.

## Positional Arguments

* `chain`: A neural network architecture, defined as either a `Flux.Chain` or a
  `Lux.AbstractLuxLayer`.
* `opt`: The optimizer to train the neural network.
* `init_params`: The initial parameter of the neural network. By default, this is `nothing`
  which thus uses the random initialization provided by the neural network library.

## Keyword Arguments

* `autodiff`: The switch between automatic (not supported yet) and numerical differentiation
              for the PDE operators. The reverse mode of the loss function is always
              automatic differentiation (via Zygote), this is only for the derivative
              in the loss function (the derivative with respect to time).
* `strategy`: The training strategy used to choose the points for the evaluations.
              By default, `GridTraining` is used with `dt` if given.
"""
@concrete struct NNDAE <: SciMLBase.AbstractDAEAlgorithm
    chain <: AbstractLuxLayer
    opt
    init_params
    autodiff::Bool
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    kwargs
end

function NNDAE(chain, opt, init_params = nothing; strategy = nothing, autodiff = false,
        kwargs...)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return NNDAE(chain, opt, init_params, autodiff, strategy, kwargs)
end

function dfdx(phi::ODEPhi, t::AbstractVector, θ, autodiff::Bool,
        differential_vars::AbstractVector)
    autodiff && throw(ArgumentError("autodiff not supported for DAE problem."))
    ϵ = sqrt(eps(eltype(t)))
    dϕ = (phi(t .+ ϵ, θ) .- phi(t, θ)) ./ ϵ
    return reduce(vcat,
        [dv ? dϕ[i:i, :] : zeros(eltype(dϕ), 1, size(dϕ, 2))
         for (i, dv) in enumerate(differential_vars)])
end

function inner_loss(phi::ODEPhi, f, autodiff::Bool, t::AbstractVector,
        θ, p, differential_vars::AbstractVector)
    out = phi(t, θ)
    dphi = dfdx(phi, t, θ, autodiff, differential_vars)
    return mapreduce(+, enumerate(t)) do (i, tᵢ)
        sum(abs2, f(dphi[:, i], out[:, i], p, tᵢ))
    end / length(t)
end

function generate_loss(strategy::GridTraining, phi::ODEPhi, f, autodiff::Bool, tspan, p,
        differential_vars::AbstractVector)
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    ts = tspan[1]:(strategy.dx):tspan[2]
    return (θ, _) -> sum(abs2, inner_loss(phi, f, autodiff, ts, θ, p, differential_vars))
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractDAEProblem,
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
        tstops = nothing
)
    (; u0, tspan, f, p, differential_vars) = prob
    t0 = tspan[1]
    (; chain, opt, autodiff, init_params) = alg

    phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
    init_params = ComponentArray(; depvar = init_params)

    @assert !isinplace(prob) "The NNODE solver only supports out-of-place DAE definitions, i.e. du=f(u,p,t)."

    strategy = if alg.strategy === nothing
        dt === nothing && error("`dt` is not defined")
        GridTraining(dt)
    end

    inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p, differential_vars)

    total_loss(θ, _) = inner_f(θ, phi)
    optf = OptimizationFunction(total_loss, AutoZygote())

    plen = maxiters === nothing ? 6 : ndigits(maxiters)
    callback = function (p, l)
        if verbose
            if maxiters === nothing
                @printf("[NNDAE]\tIter: [%*d]\tLoss: %g\n", plen, p.iter, l)
            else
                @printf("[NNDAE]\tIter: [%*d/%d]\tLoss: %g\n", plen, p.iter, maxiters, l)
            end
        end
        return l < abstol
    end

    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    # solutions at timepoints
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

    sol = SciMLBase.build_solution(prob, alg, ts, u; k = res, dense = true,
        calculate_error = false, retcode = ReturnCode.Success, original = res,
        resid = res.objective)
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    return sol
end
