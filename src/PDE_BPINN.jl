mutable struct PDELogTargetDensity{
    ST <: AbstractTrainingStrategy,
    D <: Union{Nothing, Vector{<:Matrix{<:Real}}},
    P <: Vector{<:Distribution},
    I,
    F, FF,
    PH,
}
    dim::Int64
    strategy::ST
    dataset::D
    priors::P
    allstd::Vector{Vector{Float64}}
    names::Tuple
    extraparams::Int
    init_params::I
    full_loglikelihood::F
    L2_loss2::FF
    Φ::PH

    function PDELogTargetDensity(dim, strategy, dataset,
            priors, allstd, names, extraparams,
            init_params::AbstractVector, full_loglikelihood, L2_loss2, Φ)
        new{
            typeof(strategy),
            typeof(dataset),
            typeof(priors),
            typeof(init_params),
            typeof(full_loglikelihood), typeof(L2_loss2),
            typeof(Φ),
        }(dim,
            strategy,
            dataset,
            priors,
            allstd,
            names,
            extraparams,
            init_params,
            full_loglikelihood, L2_loss2,
            Φ)
    end
    function PDELogTargetDensity(dim, strategy, dataset,
            priors, allstd, names, extraparams,
            init_params::Union{NamedTuple, ComponentArrays.ComponentVector},
            full_loglikelihood, L2_loss2, Φ)
        new{
            typeof(strategy),
            typeof(dataset),
            typeof(priors),
            typeof(init_params),
            typeof(full_loglikelihood), typeof(L2_loss2),
            typeof(Φ),
        }(dim,
            strategy,
            dataset,
            priors,
            allstd,
            names,
            extraparams,
            init_params,
            full_loglikelihood, L2_loss2,
            Φ)
    end
end

# for bc case, [bc]/bc eqs must be passed along with dataset_bc[i]
# and final loss for bc must be together in a vector(bcs has seperate type of dataset_bc)
# eqs is vector of pde eqs and dataset here is dataset_pde
# normally you get vector of losses
function get_lossy(pinnrep, dataset, Dict_differentials)
    eqs = pinnrep.eqs
    depvars = pinnrep.depvars # order is same as dataset and interps

    # Dict_differentials is filled with Differential operator => diff_i key-value pairs
    # masking operation
    eqs_new = substitute.(eqs, Ref(Dict_differentials))

    to_subs, tobe_subs = get_symbols(dataset, depvars, eqs)
    # for each row in dataset create u values for substituing in equation, n_equations=n_rows
    eq_subs = [Dict(tobe_subs[depvar] => to_subs[depvar][i] for depvar in depvars)
               for i in 1:size(dataset[1][:, 1])[1]]

    # for each point(eq_sub dictionary) substiute in all equations(eqs_new - masked equations)
    b = []
    for eq_sub in eq_subs
        push!(b, [substitute(eq, eq_sub) for eq in eqs_new])
    end
    # now we have vector of equation vectors

    # reverse dict for re-substituing values of Differential(t)(u(t)) etc
    rev_Dict_differentials = Dict(value => key for (key, value) in Dict_differentials)

    # for each vector in vecvtor of equation vectorbroadcast resubstituing OG mask values
    c = []
    for b_i in b
        push!(c, substitute.(b_i, Ref(rev_Dict_differentials)))
    end

    # get losses, zip each equation with args for each build_loss call per equation vector
    loss_functions = [[build_loss_function(pinnrep, eq, pde_indvar)
                       for (eq, pde_indvar, integration_indvar) in zip(c[i],
        pinnrep.pde_indvars,
        pinnrep.pde_integration_vars)] for i in eachindex(c)]

    return loss_functions
end

# dataset_pde has normal matrix format 
# dataset_bc has format of Vector{typeof(dataset_pde )} as each bc has different domain requirements
function get_symbols(dataset, depvars, eqs)
    depvar_vals = [dataset_i[:, 1] for dataset_i in dataset]
    # order of depvars
    to_subs = Dict(depvars .=> depvar_vals)

    asrt = Symbolics.get_variables.(eqs)
    # want only symbols of depvars
    temp = unique(reduce(vcat, asrt))

    tobe_subs = Dict()
    for a in depvars
        for i in temp
            expr = toexpr(i)
            if (expr isa Expr) && (expr.args[1] == a)
                tobe_subs[a] = i
            end
        end
    end

    return to_subs, tobe_subs
end

function LogDensityProblems.logdensity(Tar::PDELogTargetDensity, θ)
    # for parameter estimation neccesarry to use multioutput case
    return Tar.full_loglikelihood(setparameters(Tar, θ),
               Tar.allstd) + priorlogpdf(Tar, θ) + L2LossData(Tar, θ) +
           Tar.L2_loss2(setparameters(Tar, θ), Tar.allstd)
end

function setparameters(Tar::PDELogTargetDensity, θ)
    names = Tar.names
    ps_new = θ[1:(end - Tar.extraparams)]
    ps = Tar.init_params

    # multioutput case for Lux chains, for each depvar ps would contain Lux ComponentVectors
    # which we use for mapping current ahmc sampled vector of parameters onto NNs
    i = 0
    Luxparams = [vector_to_parameters(ps_new[((i += length(ps[x])) - length(ps[x]) + 1):i],
        ps[x]) for x in names]

    a = ComponentArrays.ComponentArray(NamedTuple{Tar.names}(i for i in Luxparams))

    if Tar.extraparams > 0
        b = θ[(end - Tar.extraparams + 1):end]
        return ComponentArrays.ComponentArray(;
            depvar = a,
            p = b)
    else
        return ComponentArrays.ComponentArray(;
            depvar = a)
    end
end

LogDensityProblems.dimension(Tar::PDELogTargetDensity) = Tar.dim

function LogDensityProblems.capabilities(::PDELogTargetDensity)
    LogDensityProblems.LogDensityOrder{1}()
end

# L2 losses loglikelihood(needed mainly for ODE parameter estimation)
function L2LossData(Tar::PDELogTargetDensity, θ)
    Φ = Tar.Φ
    init_params = Tar.init_params
    dataset = Tar.dataset
    sumt = 0
    L2stds = Tar.allstd[3]
    # each dep var has a diff dataset depending on its indep var and their domains
    # these datasets are matrices of first col-dep var and remaining cols-all indep var
    # Tar.init_params is needed to construct a vector of parameters into a ComponentVector

    # dataset of form Vector[matrix_x, matrix_y, matrix_z]
    # matrix_i is of form [i,indvar1,indvar2,..] (needed in case if heterogenous domains)
    # note that indvar1,indvar2.. cols can be different values for different depvar matrices
    # dataset,phi order follows pinnrep.depvars orders of variables (order of declaration in @variables macro)

    # Phi is the trial solution for each NN in chain array
    # Creating logpdf( MvNormal(Phi(t,θ),std), dataset[i] )
    # dataset[i][:, 2:end] -> indepvar cols of a particular depvar's dataset 
    # dataset[i][:, 1] -> depvar col of depvar's dataset

    if Tar.extraparams > 0
        for i in eachindex(Φ)
            sumt += logpdf(MvNormal(Φ[i](dataset[i][:, 2:end]',
                        vector_to_parameters(θ[1:(end - Tar.extraparams)],
                            init_params)[Tar.names[i]])[1,
                        :],
                    LinearAlgebra.Diagonal(abs2.(ones(size(dataset[i])[1]) .*
                                                 L2stds[i]))),
                dataset[i][:, 1])
        end
        return sumt
    end
    return 0
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
    end
    return logpdf(nnwparams, θ)
end

function inference(samples, pinnrep, saveats, numensemble, ℓπ)
    domains = pinnrep.domains
    phi = pinnrep.phi
    dict_depvar_input = pinnrep.dict_depvar_input
    depvars = pinnrep.depvars

    names = ℓπ.names
    initial_nnθ = ℓπ.init_params
    ninv = ℓπ.extraparams

    ranges = Dict([Symbol(d.variables) => infimum(d.domain):dx:supremum(d.domain)
                   for (d, dx) in zip(domains, saveats)])
    inputs = [dict_depvar_input[i] for i in depvars]

    span = [[ranges[indvar] for indvar in input] for input in inputs]
    timepoints = [hcat(vec(map(points -> collect(points),
        Iterators.product(span[i]...)))...)
                  for i in eachindex(phi)]

    # order of range's domains must match chain's inputs and dep_vars
    samples = samples[(end - numensemble):end]
    nnparams = length(samples[1][1:(end - ninv)])
    # get rows-ith param and col-ith sample value
    estimnnparams = [Particles(reduce(hcat, samples)[i, :])
                     for i in 1:nnparams]

    #  PDE params
    if ninv == 0
        estimated_params = [nothing]
    else
        estimated_params = [Particles(reduce(hcat, samples)[i, :])
                            for i in (nnparams + 1):(nnparams + ninv)]
    end

    # getting parameter ranges in case of Lux chains
    Luxparams = []
    i = 0
    for x in names
        len = length(initial_nnθ[x])
        push!(Luxparams, (i + 1):(i + len))
        i += len
    end

    # convert to format directly usable by lux
    estimatedLuxparams = [vector_to_parameters(estimnnparams[Luxparams[i]],
        initial_nnθ[names[i]]) for i in eachindex(phi)]

    # infer predictions(preds) each row - NN, each col - ith sample
    samplesn = reduce(hcat, samples)
    preds = []
    for j in eachindex(phi)
        push!(preds,
            [phi[j](timepoints[j],
                vector_to_parameters(samplesn[:, i][Luxparams[j]],
                    initial_nnθ[names[j]])) for i in 1:numensemble])
    end

    # note here no of samples referse to numensemble and points is the no of points in each dep_vars discretization
    # each phi will give output in single domain of depvar(so we have each row as a vector of vector outputs)
    # so we get after reduce a single matrix of n rows(samples), and j cols(points)
    ensemblecurves = [Particles(reduce(vcat, preds[i])) for i in eachindex(phi)]
    return ensemblecurves, estimatedLuxparams, estimated_params, timepoints
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

"""
    ahmc_bayesian_pinn_pde(pde_system, discretization;
            draw_samples = 1000,
            bcstd = [0.01], l2std = [0.05],
            phystd = [0.05], priorsNNw = (0.0, 2.0),
            param = [], nchains = 1, Kernel = HMC(0.1, 30),
            Adaptorkwargs = (Adaptor = StanHMCAdaptor,
                Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
            Integratorkwargs = (Integrator = Leapfrog,), saveats = [1 / 10.0],
            numensemble = floor(Int, draw_samples / 3), progress = false, verbose = false)               

## NOTES

* Dataset is required for accurate Parameter estimation + solving equations.
* Returned solution is a BPINNsolution consisting of Ensemble solution, estimated PDE and NN parameters
  for chosen `saveats` grid spacing and last n = `numensemble` samples in Chain. the complete set of samples
  in the MCMC chain is returned as `fullsolution`,  refer `BPINNsolution` for more details.

## Positional Arguments

* `pde_system`: ModelingToolkit defined PDE equation or system of equations.
* `discretization`: BayesianPINN discretization for the given pde_system, Neural Network and training strategy.

## Keyword Arguments

* `draw_samples`: number of samples to be drawn in the MCMC algorithms (warmup samples are ~2/3 of draw samples)
* `bcstd`: Vector of standard deviations of BPINN prediction against Initial/Boundary Condition equations.
* `l2std`: Vector of standard deviations of BPINN prediction against L2 losses/Dataset for each dependant variable of interest.
* `phystd`: Vector of standard deviations of BPINN prediction against Chosen Underlying PDE equations.
* `priorsNNw`: Tuple of (mean, std) for BPINN Network parameters. Weights and Biases of BPINN are Normal Distributions by default.
* `param`: Vector of chosen PDE's parameter's Distributions in case of Inverse problems.
* `nchains`: number of chains you want to sample.
* `Kernel`: Choice of MCMC Sampling Algorithm object HMC/NUTS/HMCDA (AdvancedHMC.jl implementations).
* `Adaptorkwargs`: `Adaptor`, `Metric`, `targetacceptancerate`. Refer: https://turinglang.org/AdvancedHMC.jl/stable/
   Note: Target percentage(in decimal) of iterations in which the proposals are accepted (0.8 by default).
* `Integratorkwargs`: `Integrator`, `jitter_rate`, `tempering_rate`. Refer: https://turinglang.org/AdvancedHMC.jl/stable/
* `saveats`: Grid spacing for each independent variable for evaluation of ensemble solution, estimated parameters.
* `numensemble`: Number of last samples to take for creation of ensemble solution, estimated parameters.
* `progress`: controls whether to show the progress meter or not.
* `verbose`: controls the verbosity. (Sample call args in AHMC).

## Warnings

* AdvancedHMC.jl is still developing convenience structs so might need changes on new releases.
"""
function ahmc_bayesian_pinn_pde(pde_system, discretization;
        draw_samples = 1000,
        bcstd = [0.01], l2std = [0.05],
        phystd = [0.05], priorsNNw = (0.0, 2.0),
        param = [], nchains = 1, Kernel = HMC(0.1, 30),
        Adaptorkwargs = (Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
        Integratorkwargs = (Integrator = Leapfrog,), saveats = [1 / 10.0],
        numensemble = floor(Int, draw_samples / 3), Dict_differentials = Dict(),
        progress = false, verbose = false)
    pinnrep = symbolic_discretize(pde_system, discretization)
    dataset_pde, dataset_bc = discretization.dataset

    yuh1 = get_lossy(pinnrep, dataset_pde, Dict_differentials)
    # eqs = pinnrep.bcs
    # yuh2 = get_lossy(pinnrep, dataset_pde, eqs)

    # this is a vector of tuple{vector,nothing}
    pde_loss_functions = [merge_strategy_with_loglikelihood_function(pinnrep::PINNRepresentation,
        GridTraining(0.1),
        yuh1[i],
        nothing; train_sets_pde = [data_pde[i, :] for data_pde in dataset_pde],
        train_sets_bc = nothing)
                          for i in eachindex(yuh1)]

    function L2_loss2(θ, allstd)
        stdpdes, stdbcs, stdextra = allstd
        # first vector of losses,from tuple -> pde losses, first[1] pde loss
        pde_loglikelihoods = [[logpdf(Normal(0, stdpdes[j]), pde_loss_function(θ))
                               for (j, pde_loss_function) in enumerate(pde_loss_functions[i][1])]
                              for i in eachindex(pde_loss_functions)]

        # bc_loglikelihoods = [logpdf(Normal(0, stdbcs[j]), bc_loss_function(θ))
        #                      for (j, bc_loss_function) in enumerate(bc_loss_functions)]

        return sum(sum(pde_loglikelihoods))
        # sum(sum(pde_loglikelihoods) + sum(bc_loglikelihoods))
    end

    # WIP split dataset to respective equations
    if ((dataset_bc isa Nothing) && (dataset_pde isa Nothing))
        dataset = nothing
    elseif dataset_bc isa Nothing
        dataset = dataset_pde
    elseif dataset_pde isa Nothing
        dataset = dataset_bc
    else
        dataset = [vcat(dataset_pde[i], dataset_bc[i]) for i in eachindex(dataset_pde)]
    end

    if discretization.param_estim && isempty(param)
        throw(UndefVarError(:param))
    elseif discretization.param_estim && dataset isa Nothing
        throw(UndefVarError(:dataset))
    elseif discretization.param_estim && length(l2std) != length(pinnrep.depvars)
        throw(error("L2 stds length must match number of dependant variables"))
    end

    # for physics loglikelihood
    full_weighted_loglikelihood = pinnrep.loss_functions.full_loss_function
    chain = discretization.chain

    if length(pinnrep.domains) != length(saveats)
        throw(error("Number of independent variables must match saveat inference discretization steps"))
    end

    # NN solutions for loglikelihood which is used for L2lossdata
    Φ = pinnrep.phi

    # for new L2 loss
    # discretization.additional_loss = 

    if nchains < 1
        throw(error("number of chains must be greater than or equal to 1"))
    end

    # remove inv params take only NN params, AHMC uses Float64
    initial_nnθ = pinnrep.flat_init_params[1:(end - length(param))]
    initial_θ = collect(Float64, initial_nnθ)

    # contains only NN parameters
    initial_nnθ = pinnrep.init_params

    names = ntuple(i -> pinnrep.depvars[i], length(chain))

    #ode parameter estimation
    nparameters = length(initial_θ)
    ninv = length(param)
    # add init_params for NN params
    priors = [
        MvNormal(priorsNNw[1] * ones(nparameters),
            LinearAlgebra.Diagonal(abs2.(priorsNNw[2] .* ones(nparameters)))),
    ]

    # append Ode params to all paramvector - initial_θ
    if ninv > 0
        # shift ode params(initialise ode params by prior means)
        # check if means or user speified is better
        initial_θ = vcat(initial_θ, [Distributions.params(param[i])[1] for i in 1:ninv])
        priors = vcat(priors, param)
        nparameters += ninv
    end

    # vector in case of N-dimensional domains
    strategy = discretization.strategy

    # dimensions would be total no of params,initial_nnθ for Lux namedTuples 
    ℓπ = PDELogTargetDensity(nparameters,
        strategy,
        dataset,
        priors,
        [phystd, bcstd, l2std],
        names,
        ninv,
        initial_nnθ,
        full_weighted_loglikelihood,
        L2_loss2,
        Φ)

    Adaptor, Metric, targetacceptancerate = Adaptorkwargs[:Adaptor],
    Adaptorkwargs[:Metric], Adaptorkwargs[:targetacceptancerate]

    # Define Hamiltonian system (nparameters ~ dimensionality of the sampling space)
    metric = Metric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    @info("Current Physics Log-likelihood : ",
        ℓπ.full_loglikelihood(setparameters(ℓπ, initial_θ),
            ℓπ.allstd))
    @info("Current Prior Log-likelihood : ", priorlogpdf(ℓπ, initial_θ))
    @info("Current MSE against dataset Log-likelihood : ", L2LossData(ℓπ, initial_θ))
    @info("Current L2_LOSSY : ",
        ℓπ.L2_loss2(setparameters(ℓπ, initial_θ),
            ℓπ.allstd))

    # parallel sampling option
    if nchains != 1

        # Cache to store the chains
        bpinnsols = Vector{Any}(undef, nchains)

        Threads.@threads for i in 1:nchains
            # each chain has different initial NNparameter values(better posterior exploration)
            initial_θ = vcat(randn(nparameters - ninv),
                initial_θ[(nparameters - ninv + 1):end])
            initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
            integrator = integratorchoice(Integratorkwargs, initial_ϵ)
            adaptor = adaptorchoice(Adaptor, MassMatrixAdaptor(metric),
                StepSizeAdaptor(targetacceptancerate, integrator))
            Kernel = AdvancedHMC.make_kernel(Kernel, integrator)
            samples, stats = sample(hamiltonian, Kernel, initial_θ, draw_samples, adaptor;
                progress = progress, verbose = verbose)

            # return a chain(basic chain),samples and stats
            matrix_samples = hcat(samples...)
            mcmc_chain = MCMCChains.Chains(matrix_samples')

            fullsolution = BPINNstats(mcmc_chain, samples, stats)
            ensemblecurves, estimnnparams, estimated_params, timepoints = inference(samples,
                pinnrep,
                saveat,
                numensemble,
                ℓπ)

            bpinnsols[i] = BPINNsolution(fullsolution,
                ensemblecurves,
                estimnnparams,
                estimated_params,
                timepoints)
        end
        return bpinnsols
    else
        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = integratorchoice(Integratorkwargs, initial_ϵ)
        adaptor = adaptorchoice(Adaptor, MassMatrixAdaptor(metric),
            StepSizeAdaptor(targetacceptancerate, integrator))

        Kernel = AdvancedHMC.make_kernel(Kernel, integrator)
        samples, stats = sample(hamiltonian, Kernel, initial_θ, draw_samples,
            adaptor; progress = progress, verbose = verbose)

        # return a chain(basic chain),samples and stats
        matrix_samples = hcat(samples...)
        mcmc_chain = MCMCChains.Chains(matrix_samples')

        @info("Sampling Complete.")
        @info("Current Physics Log-likelihood : ",
            ℓπ.full_loglikelihood(setparameters(ℓπ, samples[end]),
                ℓπ.allstd))
        @info("Current Prior Log-likelihood : ", priorlogpdf(ℓπ, samples[end]))
        @info("Current MSE against dataset Log-likelihood : ",
            L2LossData(ℓπ, samples[end]))
        @info("Current L2_LOSSY : ",
            ℓπ.L2_loss2(setparameters(ℓπ, samples[end]),
                ℓπ.allstd))

        fullsolution = BPINNstats(mcmc_chain, samples, stats)
        ensemblecurves, estimnnparams, estimated_params, timepoints = inference(samples,
            pinnrep,
            saveats,
            numensemble,
            ℓπ)

        return BPINNsolution(fullsolution,
            ensemblecurves,
            estimnnparams,
            estimated_params,
            timepoints)
    end
end