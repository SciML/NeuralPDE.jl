@concrete struct LogTargetDensity
    dim::Int
    prob <: SciMLBase.ODEProblem
    smodel <: StatefulLuxLayer
    strategy <: AbstractTrainingStrategy
    dataset <: Union{Vector{Nothing}, Vector{<:Vector{<:AbstractFloat}}}
    priors <: Vector{<:Distribution}
    phystd::Vector{Float64}
    phynewstd::Vector{Float64}
    l2std::Vector{Float64}
    autodiff::Bool
    physdt::Float64
    extraparams::Int
    init_params <: Union{NamedTuple, ComponentArray}
    estim_collocate::Bool
end

"""
NN OUTPUT AT t,θ ~ phi(t,θ).
"""
function (f::LogTargetDensity)(t::AbstractVector, θ)
    θ = vector_to_parameters(θ, f.init_params)
    dev = safe_get_device(θ)
    t = safe_expand(dev, t)
    u0 = f.prob.u0 |> dev
    return u0 .+ (t' .- f.prob.tspan[1]) .* f.smodel(t', θ)
end

(f::LogTargetDensity)(t::Number, θ) = f([t], θ)[:, 1]

"""
Similar to ode_dfdx() in NNODE.
"""
function ode_dfdx(phi::LogTargetDensity, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        return ForwardDiff.jacobian(Base.Fix2(phi, θ), t)
    else
        ϵ = sqrt(eps(eltype(t)))
        return (phi(t .+ ϵ, θ) .- phi(t, θ)) ./ ϵ
    end
end

"""
Function needed for converting vector of sampled parameters into ComponentVector in case of Lux chain output, derivatives
the sampled parameters are of exotic type `Dual` due to ForwardDiff's autodiff tagging.
"""
function vector_to_parameters(ps_new::AbstractVector, ps::Union{NamedTuple, ComponentArray})
    @assert length(ps_new) == LuxCore.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

vector_to_parameters(ps_new::AbstractVector, _::AbstractVector) = ps_new

function LogDensityProblems.logdensity(ltd::LogTargetDensity, θ)
    ldensity = physloglikelihood(ltd, θ) + priorweights(ltd, θ) + L2LossData(ltd, θ)
    ltd.estim_collocate && return ldensity + L2loss2(ltd, θ)
    return ldensity
end

LogDensityProblems.dimension(ltd::LogTargetDensity) = ltd.dim

function LogDensityProblems.capabilities(::LogTargetDensity)
    return LogDensityProblems.LogDensityOrder{1}()
end

"""
suggested extra loss function for ODE solver case
"""
@views function L2loss2(ltd::LogTargetDensity, θ)
    ltd.extraparams ≤ 0 && return false  # XXX: type-stability?

    f = ltd.prob.f
    t = ltd.dataset[end]
    u1 = ltd.dataset[2]
    û = ltd.dataset[1]

    nnsol = ode_dfdx(ltd, t, θ[1:(length(θ) - ltd.extraparams)], ltd.autodiff)

    ode_params = ltd.extraparams == 1 ? θ[((length(θ) - ltd.extraparams) + 1)] :
                 θ[((length(θ) - ltd.extraparams) + 1):length(θ)]

    physsol = if length(ltd.prob.u0) == 1
        [f(û[i], ode_params, tᵢ) for (i, tᵢ) in enumerate(t)]
    else
        [f([û[i], u1[i]], ode_params, tᵢ) for (i, tᵢ) in enumerate(t)]
    end
    # form of NN output matrix output dim x n
    deri_physsol = reduce(hcat, physsol)
    T = promote_type(eltype(deri_physsol), eltype(nnsol))

    physlogprob = T(0)
    for i in 1:length(ltd.prob.u0)
        physlogprob += logpdf(
            MvNormal(deri_physsol[i, :],
                Diagonal(abs2.(T(ltd.phynewstd[i]) .* ones(T, length(nnsol[i, :]))))),
            nnsol[i, :]
        )
    end
    return physlogprob
end

"""
L2 loss loglikelihood(needed for ODE parameter estimation).
"""
@views function L2LossData(ltd::LogTargetDensity, θ)
    (ltd.dataset isa Vector{Nothing} || ltd.extraparams == 0) && return 0

    # matrix(each row corresponds to vector u's rows)
    nn = ltd(ltd.dataset[end], θ[1:(length(θ) - ltd.extraparams)])
    T = eltype(nn)

    L2logprob = zero(T)
    for i in 1:length(ltd.prob.u0)
        # for u[i] ith vector must be added to dataset,nn[1, :] is the dx in lotka_volterra
        L2logprob += logpdf(
            MvNormal(
                nn[i, :],
                Diagonal(abs2.(T(ltd.l2std[i]) .* ones(T, length(ltd.dataset[i]))))
            ),
            ltd.dataset[i]
        )
    end
    return L2logprob
end

"""
Physics loglikelihood over problem timespan + dataset timepoints.
"""
function physloglikelihood(ltd::LogTargetDensity, θ)
    (; f, p, tspan) = ltd.prob
    (; autodiff, strategy) = ltd

    # parameter estimation chosen or not
    if ltd.extraparams > 0
        ode_params = ltd.extraparams == 1 ? θ[((length(θ) - ltd.extraparams) + 1)] :
                     θ[((length(θ) - ltd.extraparams) + 1):length(θ)]
    else
        ode_params = p isa SciMLBase.NullParameters ? Float64[] : p
    end

    return getlogpdf(strategy, ltd, f, autodiff, tspan, ode_params, θ)
end

function getlogpdf(strategy::GridTraining, ltd::LogTargetDensity, f, autodiff::Bool,
        tspan, ode_params, θ)
    ts = collect(eltype(strategy.dx), tspan[1]:(strategy.dx):tspan[2])
    t = ltd.dataset isa Vector{Nothing} ? ts : vcat(ts, ltd.dataset[end])
    return sum(innerdiff(ltd, f, autodiff, t, θ, ode_params))
end

function getlogpdf(strategy::StochasticTraining, ltd::LogTargetDensity,
        f, autodiff::Bool, tspan, ode_params, θ)
    T = promote_type(eltype(tspan[1]), eltype(tspan[2]))
    samples = (tspan[2] - tspan[1]) .* rand(T, strategy.points) .+ tspan[1]
    t = ltd.dataset isa Vector{Nothing} ? samples : vcat(samples, ltd.dataset[end])
    return sum(innerdiff(ltd, f, autodiff, t, θ, ode_params))
end

function getlogpdf(strategy::QuadratureTraining, ltd::LogTargetDensity, f, autodiff::Bool,
        tspan, ode_params, θ)
    integrand(t::Number, θ) = innerdiff(ltd, f, autodiff, [t], θ, ode_params)
    intprob = IntegralProblem(
        integrand, (tspan[1], tspan[2]), θ; nout = length(ltd.prob.u0))
    sol = solve(intprob, QuadGKJL(); strategy.abstol, strategy.reltol)
    return sum(sol.u)
end

function getlogpdf(strategy::WeightedIntervalTraining, ltd::LogTargetDensity, f,
        autodiff::Bool, tspan, ode_params, θ)
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

    t = ltd.dataset isa Vector{Nothing} ? ts : vcat(ts, ltd.dataset[end])
    return sum(innerdiff(ltd, f, autodiff, t, θ, ode_params))
end

"""
MvNormal likelihood at each `ti` in time `t` for ODE collocation residue with NN with parameters θ.
"""
@views function innerdiff(ltd::LogTargetDensity, f, autodiff::Bool, t::AbstractVector, θ,
        ode_params)
    # ltd used for phi and LogTargetDensity object attributes access
    out = ltd(t, θ[1:(length(θ) - ltd.extraparams)])

    # reject samples case(write clear reason why)
    (any(isinf, out[:, 1]) || any(isinf, ode_params)) && return convert(eltype(out), -Inf)

    # this is a vector{vector{dx,dy}}(handle case single u(float passed))
    if length(out[:, 1]) == 1
        physsol = [f(out[:, i][1], ode_params, t[i]) for i in 1:length(out[1, :])]
    else
        physsol = [f(out[:, i], ode_params, t[i]) for i in 1:length(out[1, :])]
    end
    physsol = reduce(hcat, physsol)

    nnsol = ode_dfdx(ltd, t, θ[1:(length(θ) - ltd.extraparams)], autodiff)

    vals = nnsol .- physsol
    T = eltype(vals)

    # N dimensional vector if N outputs for NN(each row has logpdf of u[i] where u is vector
    # of dependant variables)
    return [logpdf(
                MvNormal(vals[i, :],
                    Diagonal(abs2.(T(ltd.phystd[i]) .* ones(T, length(vals[i, :]))))),
                zeros(T, length(vals[i, :]))
            ) for i in 1:length(ltd.prob.u0)]
end

"""
Prior logpdf for NN parameters + ODE constants.
"""
@views function priorweights(ltd::LogTargetDensity, θ)
    allparams = ltd.priors
    nnwparams = allparams[1] # nn weights

    ltd.extraparams ≤ 0 && return logpdf(nnwparams, θ)

    # Vector of ode parameters priors
    invpriors = allparams[2:end]

    invlogpdf = sum(
        logpdf(invpriors[length(θ) - i + 1], θ[i])
    for i in (length(θ) - ltd.extraparams + 1):length(θ))

    return invlogpdf + logpdf(nnwparams, θ[1:(length(θ) - ltd.extraparams)])
end

function generate_ltd(chain::AbstractLuxLayer, init_params)
    return init_params, chain, LuxCore.initialstates(Random.default_rng(), chain)
end

function generate_ltd(chain::AbstractLuxLayer, ::Nothing)
    θ, st = LuxCore.setup(Random.default_rng(), chain)
    return θ, chain, st
end

function kernelchoice(Kernel, MCMCkwargs)
    if Kernel == HMCDA
        Kernel(MCMCkwargs[:δ], MCMCkwargs[:λ])
    elseif Kernel == NUTS
        δ, max_depth, Δ_max = MCMCkwargs[:δ], MCMCkwargs[:max_depth], MCMCkwargs[:Δ_max]
        Kernel(δ; max_depth, Δ_max)
    else # HMC
        Kernel(MCMCkwargs[:n_leapfrog])
    end
end