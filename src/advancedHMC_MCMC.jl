using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra

struct LogTargetDensity
    dim::Int
    prob::DiffEqBase.DEProblem
    re::Optimisers.Restructure
    dataset::Tuple{AbstractVector, AbstractVector}
    var::Float32
end

function LogDensityProblems.logdensity(Tar::LogTargetDensity, θ)
    return L2LossData(Tar, θ) +
           physloglikelihood(Tar, θ)
end

LogDensityProblems.dimension(Tar::LogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::Type{LogTargetDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

# nn OUTPUT AT t
function (f::LogTargetDensity)(t::AbstractVector, θ)
    vec(f.prob.u0 .+ (t .- f.prob.tspan[1]) .* f.re(θ)(adapt(parameterless_type(θ), t')))
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
    var = Tar.var
    u0 = Tar.prob.u0
    t = Tar.dataset[2]
    # let this be(will fix)
    autodiff = false

    # compare derivatives
    physsol = [f(m, p, u0) for m in Tar(t, θ)]
    nnsol = NNodederi(Tar, t, θ, autodiff)

    return loglikelihood(MvNormal(nnsol, Diagonal(var .* ones(length(nnsol)))), physsol)
end

# standard MvNormal Dist Assume
function L2LossData(Tar::LogTargetDensity, θ)
    nn = Tar.re(θ)
    return sum(abs2, (vec(nn(Tar.dataset[2]')) .- Tar.dataset[1])) ./ -2
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
function ahmc_bayesian_pinn_ode(prob::DiffEqBase.DEProblem, chain::Flux.Chain,
    dataset::Tuple{AbstractVector, AbstractVector};
    draw_samples = 500, warmup_samples = 500)
    nnparameters, recon = Flux.destructure(chain)
    nparameters = length(nnparameters)

    if isinplace(prob)
        throw(error("The BPINN ODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    # variance
    alpha = 0.09
    sig = sqrt(1.0 / alpha)

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
