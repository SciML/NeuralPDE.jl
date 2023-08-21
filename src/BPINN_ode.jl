# HIGH level API for BPINN ODE solver
struct BNNODE{C, K, IT, A, M,
              I <: Union{Nothing, Vector{<:AbstractFloat}},
              P <: Union{Vector{Nothing}, Vector{<:Distribution}},
              D <:
              Union{Vector{Nothing}, Vector{<:Vector{<:AbstractFloat}}}} <:
       NeuralPDEAlgorithm
    chain::C
    Kernel::K
    draw_samples::Int64
    priorsNNw::Tuple{Float64, Float64}
    param::P
    l2std::Vector{Float64}
    phystd::Vector{Float64}
    dataset::D
    init_params::I
    physdt::Float64
    nchains::Int64
    autodiff::Bool
    Integrator::IT
    Adaptor::A
    targetacceptancerate::Float64
    Metric::M
    jitter_rate::Float64
    tempering_rate::Float64
    max_depth::Int64
    Δ_max::Int64
    n_leapfrog::Int64
    δ::Float64
    λ::Float64
    progress::Bool
    verbose::Bool

    function BNNODE(chain, Kernel = HMC; draw_samples = 2000,
                    priorsNNw = (0.0, 2.0), param = [nothing], l2std = [0.05],
                    phystd = [0.05], dataset = [nothing],
                    init_params = nothing,
                    physdt = 1 / 20.0, nchains = 1,
                    autodiff = false, Integrator = Leapfrog,
                    Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
                    Metric = DiagEuclideanMetric, jitter_rate = 3.0,
                    tempering_rate = 3.0, max_depth = 10, Δ_max = 1000,
                    n_leapfrog = 20, δ = 0.65, λ = 0.3, progress = true,
                    verbose = false)
        new{typeof(chain), typeof(Kernel), typeof(Integrator), typeof(Adaptor),
            typeof(Metric), typeof(init_params), typeof(param),
            typeof(dataset)}(chain, Kernel, draw_samples,
                             priorsNNw, param, l2std,
                             phystd, dataset, init_params,
                             physdt, nchains, autodiff, Integrator,
                             Adaptor, targetacceptancerate,
                             Metric, jitter_rate, tempering_rate,
                             max_depth, Δ_max, n_leapfrog,
                             δ, λ, progress, verbose)
    end
end

struct BPINNstats{MC, S, ST}
    mcmc_chain::MC
    samples::S
    statistics::ST
end

struct BPINNsolution{O <: BPINNstats, E,
                     NP <: Vector{<:MonteCarloMeasurements.Particles{<:Float64}},
                     OP <: Union{Vector{Nothing},
                           Vector{<:MonteCarloMeasurements.Particles{<:Float64}}}}
    original::O
    ensemblesol::E
    estimated_nn_params::NP
    estimated_ode_params::OP

    function BPINNsolution(original, ensemblesol, estimated_nn_params, estimated_ode_params)
        new{typeof(original), typeof(ensemblesol), typeof(estimated_nn_params),
            typeof(estimated_ode_params)}(original, ensemblesol, estimated_nn_params,
                                          estimated_ode_params)
    end
end

function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return Functors.fmap(get_ps, ps)
end

function DiffEqBase.__solve(prob::DiffEqBase.ODEProblem,
                            alg::BNNODE,
                            args...;
                            dt = nothing,
                            timeseries_errors = true,
                            save_everystep = true,
                            adaptive = false,
                            abstol = 1.0f-6,
                            reltol = 1.0f-3,
                            verbose = false,
                            saveat = 1 / 50.0,
                            maxiters = nothing,
                            numensemble = 500)
    @unpack chain, l2std, phystd, param, priorsNNw, Kernel,
    draw_samples, dataset, init_params, Integrator, Adaptor, Metric,
    nchains, max_depth, Δ_max, n_leapfrog, physdt, targetacceptancerate,
    jitter_rate, tempering_rate, δ, λ, autodiff, progress, verbose = alg

    param = param == [nothing] ? [] : param

    if draw_samples < 0
        throw(error("Number of samples to be drawn has to be >=0."))
    end

    mcmcchain, samples, statistics = ahmc_bayesian_pinn_ode(prob, chain, dataset = dataset,
                                                            draw_samples = draw_samples,
                                                            init_params = init_params,
                                                            physdt = physdt, l2std = l2std,
                                                            phystd = phystd,
                                                            priorsNNw = priorsNNw,
                                                            param = param,
                                                            nchains = nchains,
                                                            autodiff = autodiff,
                                                            Kernel = Kernel,
                                                            Integrator = Integrator,
                                                            Adaptor = Adaptor,
                                                            targetacceptancerate = targetacceptancerate,
                                                            Metric = Metric,
                                                            jitter_rate = jitter_rate,
                                                            tempering_rate = tempering_rate,
                                                            max_depth = max_depth,
                                                            Δ_max = Δ_max,
                                                            n_leapfrog = n_leapfrog, δ = δ,
                                                            λ = λ, progress = progress,
                                                            verbose = verbose)

    fullsolution = BPINNstats(mcmcchain, samples, statistics)
    ninv = length(param)
    t = collect(eltype(saveat), prob.tspan[1]:saveat:prob.tspan[2])

    if chain isa Lux.AbstractExplicitLayer
        θinit, st = Lux.setup(Random.default_rng(), chain)
        θ = [vector_to_parameters(samples[i][1:(end - ninv)], θinit)
             for i in (draw_samples - numensemble):draw_samples]
        luxar = [chain(t', θ[i], st)[1] for i in 1:numensemble]

    elseif chain isa Flux.Chain
        θinit, re1 = Flux.destructure(chain)
        out = re1.([samples[i][1:(end - ninv)]
                    for i in (draw_samples - numensemble):draw_samples])
        luxar = collect(out[i](t') for i in eachindex(out))

    else
        throw(error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported"))
    end

    nnparams = length(θinit)
    ensemblecurve = [Particles(reduce(vcat, luxar)[:, i]) for i in 1:length(t)]
    estimnnparams = [Particles(reduce(hcat, samples)[i, :]) for i in 1:nnparams]

    if ninv == 0
        estimated_params = [nothing]
    else
        estimated_params = [Particles(reduce(hcat, samples[(end - ninv + 1):end])[i, :])
                            for i in (nnparams + 1):(nnparams + ninv)]
    end

    BPINNsolution(fullsolution, ensemblecurve, estimnnparams, estimated_params)
end