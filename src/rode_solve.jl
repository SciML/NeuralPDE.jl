struct NNRODE{C, W, O, P, B, K, S <: Union{Nothing, AbstractTrainingStrategy}} <:
    NeuralPDEAlgorithm
 chain::C
 W::W
 opt::O
 init_params::P
 autodiff::Bool
 batch::B
 strategy::S
 kwargs::K
end
function NNRODE(chain, W, opt, init_params = nothing;
            strategy = nothing,
            autodiff = false, batch = nothing, kwargs...)
    NNRODE(chain, W, opt, init_params, autodiff, batch, strategy, kwargs)
end

mutable struct RODEPhi{C, T, U, S}
    chain::C
    t0::T
    u0::U
    function RODEPhi(re::Optimisers.Restructure, t, u0)
        new{typeof(re), typeof(t), typeof(u0), Nothing}(re, t, u0)
    end
end

function generate_phi_θ(chain::Flux.Chain, t, u0, init_params::Nothing)
    θ, re = Flux.destructure(chain)
    RODEPhi(re, t, u0), θ
end

function generate_phi_θ(chain::Flux.Chain, t, u0, init_params)
    θ, re = Flux.destructure(chain)
    RODEPhi(re, t, u0), init_params
end

function (f::RODEPhi{C, T, U})(t::Number, w::Number,
    θ) where {C <: Optimisers.Restructure, T, U <: Number}
f.u0 + (t - f.t0) * first(f.chain(θ)(adapt(parameterless_type(θ), [t, w])))
end

function (f::RODEPhi{C, T, U})(t::AbstractVector, W::AbstractVector,
    θ) where {C <: Optimisers.Restructure, T, U <: Number}
f.u0 .+ (t' .- f.t0) .* f.chain(θ)(adapt(parameterless_type(θ), [t W]'))
end

function (f::RODEPhi{C, T, U})(t::Number, w::Number, θ) where {C <: Optimisers.Restructure, T, U}
f.u0 + (t - f.t0) * f.chain(θ)(adapt(parameterless_type(θ), [t]))
end

function (f::RODEPhi{C, T, U})(t::AbstractVector, w::AbstractVector,
    θ) where {C <: Optimisers.Restructure, T, U}
f.u0 .+ (t .- f.t0) .* f.chain(θ)(adapt(parameterless_type(θ), [t, W]'))
end

function rode_dfdx end

function rode_dfdx(phi::RODEPhi{C, T, U}, t::Number, W::Number, θ,
                  autodiff::Bool) where {C, T, U <: Number}
    if autodiff
        ForwardDiff.derivative(t -> phi(t, W, θ), t)
    else
        (phi(t + sqrt(eps(typeof(t))), W, θ) - phi(t, W, θ)) / sqrt(eps(typeof(t)))
    end
end

function rode_dfdx(phi::RODEPhi{C, T, U}, t::Number, W::Number, θ,
                  autodiff::Bool) where {C, T, U <: AbstractVector}
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, W, θ), t)
    else
        (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, W, θ)) / sqrt(eps(typeof(t)))
    end
end

function rode_dfdx(phi::RODEPhi, t::AbstractVector, W::AbstractVector, θ, autodiff::Bool)
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, W, θ), t)
    else
        (phi(t .+ sqrt(eps(eltype(t))), W, θ) - phi(t, W, θ)) ./ sqrt(eps(eltype(t)))
    end
end

function inner_loss end

function inner_loss(phi::RODEPhi{C, T, U}, f, autodiff::Bool, t::Number, W::Number, θ,
                    p) where {C, T, U <: Number}
    sum(abs2, rode_dfdx(phi, t, W, θ, autodiff) - f(phi(t, W, θ), p, t, W))
end

function inner_loss(phi::RODEPhi{C, T, U}, f, autodiff::Bool, t::AbstractVector, W::AbstractVector, θ,
                    p) where {C, T, U <: Number}
    out = phi(t, W, θ)
    fs = reduce(hcat, [f(out[i], p, t[i], W[i]) for i in 1:size(out, 2)])
    dxdtguess = Array(rode_dfdx(phi, t, W, θ, autodiff))
    sum(abs2, dxdtguess .- fs) / length(t)
end

function inner_loss(phi::RODEPhi{C, T, U}, f, autodiff::Bool, t::Number, W::Number, θ,
                    p) where {C, T, U}
    sum(abs2, rode_dfdx(phi, t, W, θ, autodiff) .- f(phi(t, W, θ), p, t, W))
end

function inner_loss(phi::RODEPhi{C, T, U}, f, autodiff::Bool, t::AbstractVector, W::AbstractVector, θ,
                    p) where {C, T, U}
    out = Array(phi(t, W, θ))
    arrt = Array(t)
    fs = reduce(hcat, [f(out[:, i], p, arrt[i], W[i]) for i in 1:size(out, 2)])
    dxdtguess = Array(rode_dfdx(phi, t, W, θ, autodiff))
    sum(abs2, dxdtguess .- fs) / length(t)
end

function generate_loss(strategy::GridTraining, phi, f, autodiff::Bool, tspan, W, p, batch)
    ts = tspan[1]:(strategy.dx):tspan[2]

    # sum(abs2,inner_loss(t,θ) for t in ts) but Zygote generators are broken
    println(typeof(W))
    function loss(θ, _)
        if batch
            sum(abs2, [inner_loss(phi, f, autodiff, ts, W[j, :], θ, p) for j in 1:size(W)[1]])
        else
            sum(abs2, [sum(abs2, [inner_loss(phi, f, autodiff, t, W[j, :][i], θ, p) for (i, t) in enumerate(ts)]) for j in 1:size(W)[1]])
        end
    end
    optf = OptimizationFunction(loss, Optimization.AutoZygote())
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractRODEProblem,
                            alg::NNRODE,
                            args...;
                            dt = nothing,
                            timeseries_errors = true,
                            save_everystep = true,
                            adaptive = false,
                            abstol = 1.0f-6,
                            reltol = 1.0f-3,
                            verbose = false,
                            saveat = nothing,
                            maxiters = nothing)
    u0 = prob.u0
    W = alg.W
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

    phi, init_params = generate_phi_θ(chain, t0, u0, init_params)
    strategy = isnothing(alg.strategy) ? GridTraining(dt) : alg.strategy
    batch = isnothing(alg.batch) ? false : alg.batch

    W_prob = NoiseProblem(W, tspan)
    W_en = EnsembleProblem(W_prob)
    W_sim = solve(W_en; dt = dt, trajectories = 100)
    W_bf = Zygote.Buffer(rand(length(W_sim), length(W_sim[1])))
    for (i, sol) in enumerate(W_sim)
        W_bf[i, :] = sol
    end
    optf = generate_loss(strategy, phi, f, autodiff::Bool, tspan, W_bf, p, batch)

    iteration = 0
    callback = function (p, l)
        iteration += 1
        verbose && println("Current loss is: $l, Iteration: $iteration")
        l < abstol
    end

    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    #solutions at timepoints
    # if saveat isa Number
    #     ts = tspan[1]:saveat:tspan[2]
    # elseif saveat isa AbstractArray
    #     ts = saveat
    # elseif dt !== nothing
    #     ts = tspan[1]:dt:tspan[2]
    # elseif save_everystep
    #     ts = range(tspan[1], tspan[2], length = 100)
    # else
    #     ts = [tspan[1], tspan[2]]
    # end

    # if u0 isa Number
    #     u = [first(phi(t, res.u)) for t in ts]
    # else
    #     u = [phi(t, res.u) for t in ts]
    # end

    # sol = DiffEqBase.build_solution(prob, alg, ts, u;
    #                                 k = res, dense = true,
    #                                 interp = NNODEInterpolation(phi, res.u),
    #                                 calculate_error = false,
    #                                 retcode = :Success)
    # DiffEqBase.has_analytic(prob.f) &&
    #     DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
    #                                           dense_errors = false)
    res, u(t, W) -> phi(t, W, res.u)
end #solve
