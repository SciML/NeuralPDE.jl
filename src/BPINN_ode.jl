# HIGH level API for BPINN ODE AND PDE SOLVER
# using MonteCarloMeasuremne
struct BNNODE{C, K, P <: Union{Vector{Nothing}, Vector{<:Distribution}}} <:
       NeuralPDEAlgorithm
    chain::C
    Kernel::K
    draw_samples::Int64
    priorsNNw::Tuple{Float64, Float64}
    param::P
    l2std::Vector{Float64}
    phystd::Vector{Float64}

    function BNNODE(chain, Kernel = HMC; draw_samples = 2000,
                    priorsNNw = (0.0, 2.0), param = [nothing], l2std = [0.05],
                    phystd = [0.05])
        new{typeof(chain), typeof(Kernel), typeof(param)}(chain,
                                                          Kernel,
                                                          draw_samples,
                                                          priorsNNw,
                                                          param, l2std,
                                                          phystd)
    end
end

struct BPINNsolution{O, E, NP <: Vector{Float64},
                     OP <: Union{Vector{Nothing}, Vector{Float64}}}
    original::O
    ensemblesol::E
    estimated_ode_params::OP
    estimated_nn_params::NP

    function BPINNsolution(original, ensemblesol, estimated_ode_params)
        new{typeof(original), typeof(ensemblesol), typeof(estimated_nn_params),
            typeof(estimated_ode_params)}
        (original, ensemblesol, estimated_nn_params, estimated_ode_params)
    end
end

struct BPINNstats{MC, S, ST}
    mcmc_chain::MC
    samples::S
    statistics::ST
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

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::BNNODE; dataset = [nothing], dt = 1 / 20.0,
                            saveat = 1 / 50.0, init_params = nothing, nchains = 1,
                            autodiff = false, Integrator = Leapfrog,
                            Adaptor = StanHMCAdaptor, targetacceptancerate = 0.8,
                            Metric = DiagEuclideanMetric, jitter_rate = 3.0,
                            tempering_rate = 3.0, max_depth = 10, Δ_max = 1000,
                            n_leapfrog = 10, δ = 0.65, λ = 0.3, progress = true,
                            verbose = false, numensemble = 500)
    chain = alg.chain
    l2std = alg.l2std
    phystd = alg.phystd
    param = alg.param == [nothing] ? [] : alg.param
    param = alg.param
    priorsNNw = alg.priorsNNw
    Kernel = alg.Kernel
    draw_samples = alg.draw_samples

    if draw_samples < 0
        throw(error("Number of samples to be drawn has to be >=0."))
    end

    mcmcchain, samples, statistics = ahmc_bayesian_pinn_ode(prob, chain, dataset = dataset,
                                                            draw_samples = draw_samples,
                                                            init_params = init_params,
                                                            physdt = dt, l2std = l2std,
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

    fullsolution = BPINNstats{MC, S, ST}(mcmcchain, samples, statistics)
    ninv = length(param)
    t = collect(eltype(saveat), prob.timespan[1]:saveat:prob.timespan[2])

    if chain isa Lux.AbstractExplicitLayer
        θinit, st = Lux.setup(Random.default_rng(), chain)
        θ = [vector_to_parameters(samples[i][1:(end - ninv)], θinit)
             for i in (draw_samples - numensemble):draw_samples]
        luxar = [chain(t', θ[i], st)[1] for i in 1:numensemble]

    elseif chain isa Flux.Chain
        θinit, re1 = destructure(chain)
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
        estimated_odeparams = Float64[]
        estimodeparam = [Particles(reduce(hcat, samples[(end - ninv + 1):end])[i, :])
                         for i in 1:nnparams]

        for j in 1:ninv
            push!(estimated_params,
                  mean([samples[i][end - ninv + j]
                        for i in (draw_samples - numensemble):draw_samples]))
        end
    end

    BPINNsolution{O, E}(fullsolution, ensemblecurve, estimnnparams, estimated_params)
end