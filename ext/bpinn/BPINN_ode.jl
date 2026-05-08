# HIGH level API for BPINN ODE solver — convenience constructor + solve method.
# The BNNODE / BPINNstats / BPINNsolution structs themselves are defined in
# `src/bpinn_types.jl` so that they remain visible without loading the extension.

function NeuralPDE.BNNODE(
        chain, kernel = HMC; strategy = nothing, draw_samples = 1000,
        priorsNNw = (0.0, 2.0), param = nothing, l2std = [0.05], phystd = [0.05],
        phynewstd = (ode_params) -> [0.05], dataset = [], physdt = 1 / 20.0,
        MCMCkwargs = (n_leapfrog = 30,), nchains = 1, init_params = nothing,
        Adaptorkwargs = (
            Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8,
        ),
        Integratorkwargs = (Integrator = Leapfrog,),
        numensemble = floor(Int, draw_samples / 3),
        estim_collocate = false, autodiff = false, progress = false, verbose = false
    )
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return BNNODE(
        chain, kernel, strategy, draw_samples, priorsNNw, param, l2std, phystd,
        phynewstd, dataset, physdt, MCMCkwargs, nchains, init_params, Adaptorkwargs,
        Integratorkwargs, numensemble, estim_collocate, autodiff, progress, verbose
    )
end

function SciMLBase.__solve(
        prob::SciMLBase.ODEProblem, alg::BNNODE, args...; dt = nothing,
        timeseries_errors = true, save_everystep = true, adaptive = false,
        abstol = 1.0f-6, reltol = 1.0f-3, verbose = false, saveat = 1 / 50.0,
        maxiters = nothing
    )
    (; chain, param, strategy, draw_samples, numensemble, verbose) = alg

    # ahmc_bayesian_pinn_ode needs param=[] for easier vcat operation for full vector of parameters
    param = param === nothing ? [] : param
    strategy = strategy === nothing ? GridTraining : strategy

    @assert alg.draw_samples ≥ 0 "Number of samples to be drawn has to be >=0."

    mcmcchain, samples,
        statistics = NeuralPDE.ahmc_bayesian_pinn_ode(
        prob, chain; strategy, alg.dataset, alg.draw_samples, alg.init_params,
        alg.physdt, alg.l2std, alg.phystd, alg.phynewstd,
        alg.priorsNNw, param, alg.nchains, alg.autodiff,
        Kernel = alg.kernel, alg.Adaptorkwargs, alg.Integratorkwargs,
        alg.MCMCkwargs, alg.progress, alg.verbose, alg.estim_collocate
    )

    fullsolution = BPINNstats(mcmcchain, samples, statistics)
    ninv = length(param)
    t = collect(eltype(saveat), prob.tspan[1]:saveat:prob.tspan[2])

    θinit, st = LuxCore.setup(Random.default_rng(), chain)
    θ = [
        vector_to_parameters(samples[i][1:(end - ninv)], θinit)
            for i in (draw_samples - numensemble):draw_samples
    ]

    luxar = [chain(t', θ[i], st)[1] for i in 1:numensemble]
    # only need for size
    θinit = collect(ComponentArray(θinit))

    # constructing ensemble predictions
    ensemblecurves = Vector{}[]
    # check if NN output is more than 1
    numoutput = size(luxar[1])[1]
    if numoutput > 1
        # Initialize a vector to store the separated outputs for each output dimension
        output_matrices = [Vector{Vector{Float32}}() for _ in 1:numoutput]

        # Loop through each element in `luxar`
        for element in luxar
            for i in 1:numoutput
                push!(output_matrices[i], element[i, :])  # Append the i-th output (i-th row) to the i-th output_matrices
            end
        end

        for r in 1:numoutput
            ensem_r = hcat(output_matrices[r]...)'
            ensemblecurve_r = prob.u0[r] .+
                [Particles(ensem_r[:, i]) for i in 1:length(t)] .*
                (t .- prob.tspan[1])
            push!(ensemblecurves, ensemblecurve_r)
        end

    else
        ensemblecurve = prob.u0 .+
            [Particles(reduce(vcat, luxar)[:, i]) for i in 1:length(t)] .*
            (t .- prob.tspan[1])
        push!(ensemblecurves, ensemblecurve)
    end

    nnparams = length(θinit)
    estimnnparams = [
        Particles(reduce(hcat, samples[(end - numensemble):end])[i, :])
            for i in 1:nnparams
    ]

    if ninv == 0
        estimated_params = [nothing]
    else
        estimated_params = [
            Particles(reduce(hcat, samples[(end - numensemble):end])[i, :])
                for i in (nnparams + 1):(nnparams + ninv)
        ]
    end

    return BPINNsolution(fullsolution, ensemblecurves, estimnnparams, estimated_params, t)
end
