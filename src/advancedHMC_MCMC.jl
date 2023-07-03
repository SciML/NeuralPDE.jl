using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra

struct LogTargetDensity
    dim::Int
    prob::DiffEqBase.DEProblem
    re::Optimisers.Restructure
    dataset::Tuple{AbstractVector, AbstractVector}
    var::Tuple{Float64, Float64}
end

function LogDensityProblems.logdensity(Tar::LogTargetDensity, θ)
    alpha = 0.1
    return physloglikelihood(Tar, θ)
    #  + L2LossData(Tar, θ)

    # + priorweights(Tar, θ)
end

LogDensityProblems.dimension(Tar::LogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::Type{LogTargetDensity})
    LogDensityProblems.LogDensityOrder{1}()
end

# nn OUTPUT AT t
function (f::LogTargetDensity)(t::AbstractVector, θ)
    f.prob.u0 .+ (t .- f.prob.tspan[1]) .* vec(f.re(θ)(t'))
end

# ODE DU/DX
function NNodederi(phi::LogTargetDensity, t::AbstractVector, θ, autodiff::Bool)
    if autodiff
        ForwardDiff.jacobian(t -> phi(t, θ), t)
    else
        (phi(t .+ sqrt(eps(eltype(t))), θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
    end
end

function physloglikelihood(Tar::LogTargetDensity, θ)
    p = Tar.prob.p
    f = Tar.prob.f
    var = 0.05^2
    u0 = Tar.prob.u0
    t = Tar.dataset[2]
    # distributions cannot take in forwarddiff jacobian matrices as means
    autodiff = true
    # compare derivatives
    out = Tar(t, θ)
    # print(size(out))
    physsol = [f(out[i], p, t[i]) for i in eachindex(out)]
    # print(size(physsol))
    nnsol = NNodederi(Tar, t, θ, autodiff)
    if autodiff
        nnsol = diag(nnsol)
    end

    n = length(nnsol)

    # # return sum(abs2, (nnsol .- physsol)) ./ (-2 * (var^2))
    # # return ((-n / 2 * log(2π * var^2)) - (sum(abs2, (nnsol .- physsol)) / (2 * var^2)))
    return logpdf(MvNormal(nnsol, Diagonal(var .* ones(n))), physsol)
    # # return loglikelihood(MvNormal(nnsol, Diagonal(var .* ones(length(nnsol)))), physsol)
end

# standard MvNormal Dist Assume
function L2LossData(Tar::LogTargetDensity, θ)
    nn = vec(Tar.re(θ)(Tar.dataset[2]'))
    n = length(nn)
    var = 0.05^2
    # return loglikelihood(MvNormal(nn, Diagonal(Tar.var .* ones(length(nn)))),Tar.dataset[1])
    return logpdf(MvNormal(nn, Diagonal(var .* ones(n))), Tar.dataset[1])

    # return ((-n / 2 * log(2π * var^2)) - (sum(abs2, (nn .- Tar.dataset[1])) / (2 * var^2)))
end

function priorweights(Tar::LogTargetDensity, θ)
    params = Tar.var
    return logpdf(MvNormal(θ, Diagonal(params[2]^2 .* ones(length(θ)))),
                  params[1] * ones(length(θ)))
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
function ahmc_bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain::Flux.Chain,
                                dataset::Tuple{AbstractVector, AbstractVector};
                                draw_samples = 1000, warmup_samples = 1000)
    nnparameters, recon = Flux.destructure(chain)
    nparameters = length(nnparameters)

    if isinplace(prob)
        throw(error("The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    # variance
    varsd = 2.0
    varμ = 0.0
    sig = (varμ, varsd)
    # sig = eps(eltype(dataset[1][1]))

    initial_θ = collect(Float64, vec(nnparameters))
    ℓπ = LogTargetDensity(nparameters, prob, recon, dataset,
                          sig)
    n_samples, n_adapts = draw_samples, warmup_samples
    metric = DiagEuclideanMetric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)
    proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    samples, stats = sample(hamiltonian, proposal, initial_θ,
                            n_samples, adaptor, n_adapts;
                            progress = true)
    return samples, stats
end

# non vectorised functions(i noticed sampling time increase)
# function NNodederi(phi::odeByNN, t::Number, θ, autodiff::Bool)
#     if autodiff
#         ForwardDiff.jacobian(t -> phi(t, θ), t)
#     else
#         (phi(t + sqrt(eps(typeof(t))), θ) - phi(t, θ)) / sqrt(eps(typeof(t)))
#     end
# end

# function (f::odeByNN{C, T, U})(t::Number,
#                                θ) where {C <: Optimisers.Restructure, T, U}
#     f.u0 + (t - f.t0) * first(f.chain(θ)(adapt(parameterless_type(θ), [t])))
# end
