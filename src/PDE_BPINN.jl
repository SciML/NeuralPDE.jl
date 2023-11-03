mutable struct PDELogTargetDensity{
    ST <: AbstractTrainingStrategy,
    D <: Union{Vector{Nothing}, Vector{<:Vector{<:AbstractFloat}}},
    P <: Vector{<:Distribution},
    I,
    F,
    PH,
}
    dim::Int64
    strategy::ST
    dataset::D
    priors::P
    allstd::Vector{Vector{Float64}}
    autodiff::Bool
    physdt::Float64
    extraparams::Int
    init_params::I
    full_loglikelihood::F
    Phi::PH

    function PDELogTargetDensity(dim, strategy, dataset,
        priors, allstd, autodiff, physdt, extraparams,
        init_params::AbstractVector, full_loglikelihood, Phi)
        new{
            typeof(strategy),
            typeof(dataset),
            typeof(priors),
            typeof(init_params),
            typeof(full_loglikelihood),
            typeof(Phi),
        }(dim,
            strategy,
            dataset,
            priors,
            allstd,
            autodiff,
            physdt,
            extraparams,
            init_params,
            full_loglikelihood,
            Phi)
    end
    function PDELogTargetDensity(dim, strategy, dataset,
        priors, allstd, autodiff, physdt, extraparams,
        init_params::NamedTuple, full_loglikelihood, Phi)
        new{
            typeof(strategy),
            typeof(dataset),
            typeof(priors),
            typeof(init_params),
            typeof(full_loglikelihood),
            typeof(Phi),
        }(dim,
            strategy,
            dataset,
            priors,
            allstd,
            autodiff,
            physdt,
            extraparams,
            init_params,
            full_loglikelihood,
            Phi)
    end
end

function LogDensityProblems.logdensity(Tar::PDELogTargetDensity, θ)
    # forward solving 
    # Tar.full_loglikelihood(vector_to_parameters(θ, Tar.init_params), Tar.allstd)
    # println("1 : ",
    #     length(Tar.full_loglikelihood(vector_to_parameters(θ,
    #             Tar.init_params),
    #         Tar.allstd).partials))
    # println("2 : ", L2LossData(Tar, θ).value)
    # println("2 : ", L2LossData(Tar, θ).partials)

    # # println("3 : ", length(priorlogpdf(Tar, θ).partials))

    # # println("sum : ",
    # #     (Tar.full_loglikelihood(vcat(vector_to_parameters(θ[1:(end - 1)],
    # #                  Tar.init_params[1:(end - 1)]), θ[end]),
    # #          Tar.allstd) +
    # #      L2LossData(Tar, θ) + priorlogpdf(Tar, θ)).value)
    # println(typeof(θ) <: AbstractVector)
    # println(length(θ))

    # println("1 : ",
    #     length(Tar.full_loglikelihood(vcat(vector_to_parameters(θ[1:(end - Tar.extraparams)],
    #                 Tar.init_params[1:(end - Tar.extraparams)]), θ[(end - Tar.extraparams + 1):end]),
    #         Tar.allstd).partials))
    # println("2 : ", length(L2LossData(Tar, θ).partials))
    # println("3 : ", length(priorlogpdf(Tar, θ).partials))

    return Tar.full_loglikelihood(vcat(vector_to_parameters(θ[1:(end - Tar.extraparams)],
                       Tar.init_params[1:(end - Tar.extraparams)]), θ[(end - Tar.extraparams + 1):end]),
               Tar.allstd) +
           L2LossData(Tar, θ) + priorlogpdf(Tar, θ)
end

LogDensityProblems.dimension(Tar::PDELogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::PDELogTargetDensity)
    LogDensityProblems.LogDensityOrder{1}()
end

# L2 losses loglikelihood(needed mainly for ODE parameter estimation)
function L2LossData(Tar::PDELogTargetDensity, θ)
    # matrix(each row corresponds to vector u's rows)
    if Tar.dataset isa Vector{Nothing} || Tar.extraparams == 0
        return 0
    else
        nn = [phi(Tar.dataset[end]', θ[1:(length(θ) - Tar.extraparams)])
              for phi in Tar.Phi]

        L2logprob = 0
        for i in 1:(length(Tar.dataset) - 1)
            # for u[i] ith vector must be added to dataset,nn[1,:] is the dx in lotka_volterra
            L2logprob += logpdf(MvNormal(nn[i][:],
                    ones(length(Tar.dataset[end])) .* Tar.allstd[3]),
                Tar.dataset[i])
        end

        return L2logprob
    end
end

# priors for NN parameters + ODE constants
function priorlogpdf(Tar::PDELogTargetDensity, θ)
    allparams = Tar.priors
    # Vector of ode parameters priors
    invpriors = allparams[2:end]

    # nn weights
    nnwparams = allparams[1]

    if Tar.extraparams > 0
        invlogpdf = sum(logpdf(invpriors[length(θ) - i + 1], θ[i])
                        for i in (length(θ) - Tar.extraparams + 1):length(θ); init = 0.0)

        return (invlogpdf
                +
                logpdf(nnwparams, θ[1:(length(θ) - Tar.extraparams)]))
    else
        return logpdf(nnwparams, θ)
    end
end

function kernelchoice(Kernel, MCMCkwargs)
    if Kernel == HMCDA
        δ, λ = MCMCkwargs[:δ], MCMCkwargs[:λ]
        Kernel(δ, λ)
    elseif Kernel == NUTS
        δ, max_depth, Δ_max = MCMCkwargs[:δ], MCMCkwargs[:max_depth], MCMCkwargs[:Δ_max]
        Kernel(δ, max_depth = max_depth, Δ_max = Δ_max)
    else
        # HMC
        n_leapfrog = MCMCkwargs[:n_leapfrog]
        Kernel(n_leapfrog)
    end
end

function integratorchoice(Integratorkwargs, initial_ϵ)
    Integrator = Integratorkwargs[:Integrator]
    if Integrator == JitteredLeapfrog
        jitter_rate = Integratorkwargs[:jitter_rate]
        Integrator(initial_ϵ, jitter_rate)
    elseif Integrator == TemperedLeapfrog
        tempering_rate = Integratorkwargs[:tempering_rate]
        Integrator(initial_ϵ, tempering_rate)
    else
        Integrator(initial_ϵ)
    end
end

function adaptorchoice(Adaptor, mma, ssa)
    if Adaptor != AdvancedHMC.NoAdaptation()
        Adaptor(mma, ssa)
    else
        AdvancedHMC.NoAdaptation()
    end
end

# dataset would be (x̂,t)
# priors: pdf for W,b + pdf for ODE params
# lotka specific kwargs here
function ahmc_bayesian_pinn_pde(pde_system, discretization;
    strategy = GridTraining, dataset = [nothing],
    init_params = nothing, draw_samples = 1000,
    physdt = 1 / 20.0, bcstd = [0.01], l2std = [0.05],
    phystd = [0.05], priorsNNw = (0.0, 2.0),
    param = [], nchains = 1, autodiff = false,
    Kernel = HMC,
    Adaptorkwargs = (Adaptor = StanHMCAdaptor,
        Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
    Integratorkwargs = (Integrator = Leapfrog,),
    MCMCkwargs = (n_leapfrog = 30,),
    progress = false, verbose = false)
    pinnrep = symbolic_discretize(pde_system, discretization, bayesian = true)

    # for physics loglikelihood
    full_weighted_loglikelihood = pinnrep.loss_functions.full_loss_function
    chain = discretization.chain

    # NN solutions for loglikelihood which is used for L2lossdata
    Phi = pinnrep.phi

    # for new L2 loss
    # discretization.additional_loss = 

    # remove inv params
    initial_nnθ = pinnrep.flat_init_params[1:(end - length(param))]

    if nchains > Threads.nthreads()
        throw(error("number of chains is greater than available threads"))
    elseif nchains < 1
        throw(error("number of chains must be greater than 1"))
    end

    if chain isa Lux.AbstractExplicitLayer
        # Lux chain(using component array later as vector_to_parameter need namedtuple,AHMC uses Float64)
        initial_θ = collect(Float64, vcat(ComponentArrays.ComponentArray(initial_nnθ)))
        # namedtuple form of Lux params required for RuntimeGeneratedFunctions
        initial_nnθ, st = Lux.setup(Random.default_rng(), chain)
    else
        # flat_init_params contains also inv params
        # initial_θ = collect(Float64, initial_nnθ[1:(length(initial_nnθ) - length(param))])
        initial_θ = collect(Float64, initial_nnθ)
    end

    # adding ode parameter estimation
    nparameters = length(initial_θ)

    # println(Tar.Phi(initial_θ))

    ninv = length(param)
    priors = [MvNormal(priorsNNw[1] * ones(nparameters), priorsNNw[2] * ones(nparameters))]

    # append Ode params to all paramvector
    if ninv > 0
        # shift ode params(initialise ode params by prior means)
        # check if means or user speified is better
        initial_θ = vcat(initial_θ, [Distributions.params(param[i])[1] for i in 1:ninv])
        priors = vcat(priors, param)
        nparameters += ninv
    end

    strategy = strategy(physdt)

    # dimensions would be total no of params,initial_nnθ for Lux namedTuples 
    ℓπ = PDELogTargetDensity(nparameters,
        strategy,
        dataset,
        priors,
        [phystd, bcstd, l2std],
        autodiff,
        physdt,
        ninv,
        initial_nnθ,
        full_weighted_loglikelihood,
        Phi)

    println(ℓπ.full_loglikelihood(initial_θ, ℓπ.allstd))
    println(priorlogpdf(ℓπ, initial_θ))
    println(L2LossData(ℓπ, initial_θ))

    Adaptor, Metric, targetacceptancerate = Adaptorkwargs[:Adaptor],
    Adaptorkwargs[:Metric], Adaptorkwargs[:targetacceptancerate]

    # Define Hamiltonian system (nparameters ~ dimensionality of the sampling space)
    metric = Metric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    # parallel sampling option
    if nchains != 1
        # Cache to store the chains
        chains = Vector{Any}(undef, nchains)
        statsc = Vector{Any}(undef, nchains)
        samplesc = Vector{Any}(undef, nchains)

        Threads.@threads for i in 1:nchains
            # each chain has different initial NNparameter values(better posterior exploration)
            initial_θ = vcat(randn(nparameters - ninv),
                initial_θ[(nparameters - ninv + 1):end])
            initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
            integrator = integratorchoice(Integratorkwargs, initial_ϵ)
            adaptor = adaptorchoice(Adaptor, MassMatrixAdaptor(metric),
                StepSizeAdaptor(targetacceptancerate, integrator))

            MCMC_alg = kernelchoice(Kernel, MCMCkwargs)
            Kernel = AdvancedHMC.make_kernel(MCMC_alg, integrator)
            samples, stats = sample(hamiltonian, Kernel, initial_θ, draw_samples, adaptor;
                progress = progress, verbose = verbose)

            samplesc[i] = samples
            statsc[i] = stats
            mcmc_chain = Chains(hcat(samples...)')
            chains[i] = mcmc_chain
        end

        return chains, samplesc, statsc
    else
        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = integratorchoice(Integratorkwargs, initial_ϵ)
        adaptor = adaptorchoice(Adaptor, MassMatrixAdaptor(metric),
            StepSizeAdaptor(targetacceptancerate, integrator))

        MCMC_alg = kernelchoice(Kernel, MCMCkwargs)
        Kernel = AdvancedHMC.make_kernel(MCMC_alg, integrator)
        samples, stats = sample(hamiltonian, Kernel, initial_θ, draw_samples,
            adaptor; progress = progress, verbose = verbose)

        # return a chain(basic chain),samples and stats
        matrix_samples = hcat(samples...)
        mcmc_chain = MCMCChains.Chains(matrix_samples')
        return mcmc_chain, samples, stats
    end
end