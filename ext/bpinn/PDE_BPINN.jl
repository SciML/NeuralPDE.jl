"""
Bayesian PINN sampler for `PDESystem`s on the NeuralPDE 7 System pipeline.

Each residual block of the discretized `System` becomes a Gaussian likelihood over its
collocation batch; priors cover the network parameter unknowns and (when
`param_estim = true`) the estimated PDE parameters. Sampling reuses the AdvancedHMC
plumbing shared with `ahmc_bayesian_pinn_ode`.
"""

@concrete struct PDELogTargetDensity
    dim::Int
    prob <: SciMLBase.OptimizationProblem
    residual_syms
    stds::Vector{Float64}
    priors <: Vector{<:Distribution}
    extraparams::Int
    dataset
    l2std::Vector{Float64}
    network_fns
    net_lengths::Vector{Int}
end

function LogDensityProblems.logdensity(ltd::PDELogTargetDensity, θ)
    return physics_loglikelihood(ltd, θ) + priorlogpdf(ltd, θ) + L2LossData(ltd, θ)
end

LogDensityProblems.dimension(ltd::PDELogTargetDensity) = ltd.dim

function LogDensityProblems.capabilities(::PDELogTargetDensity)
    return LogDensityProblems.LogDensityOrder{1}()
end

"""
Gaussian likelihood of pointwise residuals `r ~ N(0, σ²)` for one residual batch.
"""
function residual_logpdf(r, σ::Real)
    rvec = vec(r)
    T = eltype(rvec)
    n = length(rvec)
    (any(!isfinite, rvec) || !(isfinite(σ) && σ > zero(σ))) && return T(-Inf)
    return logpdf(
        MvNormal(rvec, Diagonal(abs2.(T(σ) .* ones(T, n)))),
        zeros(T, n)
    )
end

function physics_loglikelihood(ltd::PDELogTargetDensity, θ)
    # `remake` + `getu` is ForwardDiff-compatible for array unknowns of the Systems pipeline.
    pθ = remake(ltd.prob; u0 = θ)
    ll = zero(eltype(θ))
    for (i, rsym) in enumerate(ltd.residual_syms)
        ll += residual_logpdf(getu(pθ, rsym)(pθ), ltd.stds[i])
    end
    return ll
end

@views function priorlogpdf(ltd::PDELogTargetDensity, θ)
    nnwparams = ltd.priors[1]
    ltd.extraparams ≤ 0 && return logpdf(nnwparams, θ)
    invpriors = ltd.priors[2:end]
    invlogpdf = sum(
        logpdf(invpriors[j], θ[end - ltd.extraparams + j]) for j in 1:(ltd.extraparams)
    )
    return invlogpdf + logpdf(nnwparams, θ[1:(end - ltd.extraparams)])
end

@views function L2LossData(ltd::PDELogTargetDensity, θ)
    dataset = ltd.dataset
    (dataset === nothing || ltd.extraparams ≤ 0) && return zero(eltype(θ))
    T = eltype(θ)
    sumt = zero(T)
    nnθ = θ[1:(end - ltd.extraparams)]
    offsets = _net_offsets(ltd.net_lengths)
    for (i, Φ) in enumerate(ltd.network_fns)
        X = transpose(dataset[i][:, 2:end])
        y = dataset[i][:, 1]
        θi = nnθ[(offsets[i] + 1):offsets[i + 1]]
        pred = vec(Φ(collect(X), θi))
        sumt += logpdf(
            MvNormal(pred, Diagonal(abs2.(T(ltd.l2std[i]) .* ones(T, length(y))))),
            T.(y)
        )
    end
    return sumt
end

_net_offsets(lengths) = cumsum(vcat(0, lengths))

function _block_stds(blocks, phystd, bcstd)
    pde_i = 0
    bc_i = 0
    stds = Float64[]
    for b in blocks
        if b.kind == :pde
            pde_i += 1
            pde_i <= length(phystd) || error(
                "`phystd` length $(length(phystd)) is shorter than the number of PDE equations"
            )
            push!(stds, phystd[pde_i])
        elseif b.kind == :bc
            bc_i += 1
            bc_i <= length(bcstd) || error(
                "`bcstd` length $(length(bcstd)) is shorter than the number of boundary conditions"
            )
            push!(stds, bcstd[bc_i])
        else
            error("Unknown residual block kind $(b.kind)")
        end
    end
    return stds
end

function _merge_dataset(dataset_pde, dataset_bc)
    if dataset_bc === nothing && dataset_pde === nothing
        return nothing
    elseif dataset_bc === nothing
        return dataset_pde
    elseif dataset_pde === nothing
        return dataset_bc
    else
        return [vcat(dataset_pde[i], dataset_bc[i]) for i in eachindex(dataset_pde)]
    end
end

function _saveat_grid(md, saveats)
    ivs = collect(get_ivs(md.pdesys))
    length(ivs) == length(saveats) || error(
        "Number of independent variables must match `saveats` inference discretization steps"
    )
    domains = get_domain(md.pdesys)
    dom_of = Dict{Any, Any}()
    for d in domains
        vars = unwrap(d.variables)
        vars = iscall(vars) && operation(vars) === tuple ? arguments(vars) : (vars,)
        for v in vars
            dom = d.domain
            lo = hasproperty(dom, :left) ? dom.left : DomainSets.infimum(dom)
            hi = hasproperty(dom, :right) ? dom.right : DomainSets.supremum(dom)
            dom_of[unwrap(v)] = (lo, hi)
        end
    end
    ranges = [
        begin
            lo, hi = dom_of[unwrap(iv)]
            range(float(lo), float(hi); step = float(saveats[i]))
        end
            for (i, iv) in enumerate(ivs)
    ]
    return ranges, ivs
end

function _product_coords(grids)
    return reduce(hcat, collect.(Iterators.product(grids...)))
end

function inference(samples, md, saveats, numensemble, ℓπ)
    ranges, ivs = _saveat_grid(md, saveats)
    nets = unique_networks(md.networks)
    ninv = ℓπ.extraparams
    samples = samples[(end - numensemble):end]
    nnparams = length(samples[1]) - ninv
    estimnnparams = [Particles(reduce(hcat, samples)[i, :]) for i in 1:nnparams]
    estimated_params = if ninv == 0
        [nothing]
    else
        [Particles(reduce(hcat, samples)[i, :]) for i in (nnparams + 1):(nnparams + ninv)]
    end

    ensemblecurves = []
    timepoints = []
    offsets = _net_offsets(ℓπ.net_lengths)
    samplesn = reduce(hcat, samples)
    for net in md.networks
        arg_grids = [
            begin
                i = findfirst(iv -> isequal(unwrap(iv), unwrap(a)), ivs)
                ranges[i]
            end
                for a in net.args
        ]
        tp = _product_coords(arg_grids)
        push!(timepoints, tp)
        net_idx = findfirst(n -> isequal(unwrap(n.θ), unwrap(net.θ)), nets)
        Φ = ℓπ.network_fns[net_idx]
        k = net.output
        preds = [
            Φ(tp, samplesn[:, i][(offsets[net_idx] + 1):offsets[net_idx + 1]])[k:k, :]
                for i in 1:numensemble
        ]
        push!(ensemblecurves, Particles(reduce(vcat, preds)))
    end
    estimatedLuxparams = [
        estimnnparams[(offsets[i] + 1):offsets[i + 1]]
            for i in eachindex(nets)
    ]
    return ensemblecurves, estimatedLuxparams, estimated_params, timepoints
end

"""
    ahmc_bayesian_pinn_pde(pde_system, discretization;
        draw_samples = 1000, bcstd = [0.01], l2std = [0.05], phystd = [0.05],
        priorsNNw = (0.0, 2.0), param = [], nchains = 1,
        Kernel = HMC(0.1, 30), Adaptorkwargs = (Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8),
        Integratorkwargs = (Integrator = Leapfrog,), saveats = [1 / 10.0],
        numensemble = floor(Int, draw_samples / 3), progress = false, verbose = false)

Bayesian inference for a ModelingToolkit `PDESystem` using the NeuralPDE 7
`PhysicsInformedNN` / `BayesianPINN` pipeline.

The log-density is built from the discretized `System`: each residual block is a
Gaussian likelihood over its collocation batch with noise std taken from `phystd`
(PDE equations) or `bcstd` (boundary conditions). Priors cover the array unknowns
and, when `param_estim = true`, the estimated PDE parameters in `param`.

## Positional Arguments

* `pde_system`: ModelingToolkit `PDESystem`.
* `discretization`: a [`BayesianPINN`](@ref) or [`PhysicsInformedNN`](@ref).

## Keyword Arguments

* `draw_samples`: number of MCMC samples (warmup is ~2/3 of this).
* `bcstd`: noise std of each boundary-condition residual batch.
* `phystd`: noise std of each PDE residual batch.
* `l2std`: noise std of the observational L2 likelihood (inverse problems).
* `priorsNNw`: `(mean, std)` of the isotropic Normal prior on network weights.
* `param`: prior distributions of estimated PDE parameters (`param_estim = true`).
* `nchains`: number of MCMC chains.
* `Kernel`, `Adaptorkwargs`, `Integratorkwargs`: AdvancedHMC sampling controls.
* `saveats`: grid spacing per independent variable for the ensemble solution.
* `numensemble`: trailing samples used for the ensemble / parameter estimates.
* `pretrain_iters`: number of Adam steps on the `OptimizationProblem` objective used to
  warm-start the MCMC chain (default `500`). Set to `0` to sample from the Lux
  initialization directly.
* `progress`, `verbose`: AdvancedHMC verbosity.

Returns a [`BPINNsolution`](@ref) (or a vector of them when `nchains > 1`).
"""
function NeuralPDE.ahmc_bayesian_pinn_pde(
        pde_system, discretization;
        draw_samples = 1000, bcstd = [0.01], l2std = [0.05], phystd = [0.05],
        phynewstd = [0.05], priorsNNw = (0.0, 2.0), param = [], nchains = 1,
        Kernel = HMC(0.1, 30), Adaptorkwargs = (
            Adaptor = StanHMCAdaptor,
            Metric = DiagEuclideanMetric, targetacceptancerate = 0.8,
        ),
        Integratorkwargs = (Integrator = Leapfrog,), saveats = [1 / 10.0],
        numensemble = floor(Int, draw_samples / 3), Dict_differentials = nothing,
        pretrain_iters::Int = 500, progress = false, verbose = false
    )
    if Dict_differentials !== nothing
        @warn """
        `Dict_differentials` (data-quadrature / operator-masking likelihood) is not
        reconstructed on the NeuralPDE 7 Systems pipeline yet; sampling proceeds with
        physics, boundary, prior and L2-data terms only. `phynewstd` is ignored.
        """
    end

    pinn = discretization isa BayesianPINN ? discretization.pinn : discretization
    dataset_pde, dataset_bc = if discretization isa BayesianPINN
        discretization.dataset
    else
        (nothing, nothing)
    end
    dataset = _merge_dataset(dataset_pde, dataset_bc)

    prob = SciMLBase.discretize(pde_system, pinn; adtype = AutoForwardDiff())
    md = pinn_metadata(prob)

    if pinn.param_estim && isempty(param)
        throw(UndefVarError(:param))
    elseif pinn.param_estim && dataset === nothing
        throw(UndefVarError(:dataset))
    elseif pinn.param_estim && length(l2std) != length(md.networks)
        error("L2 stds length must match number of dependent variables")
    end

    nets = unique_networks(md.networks)
    net_lengths = Int[length(getdefault(net.θ)) for net in nets]
    n_nn = sum(net_lengths)
    ninv = length(param)

    # OptimizationProblem stores array unknowns as one flat `Vector`.
    initial_θ = collect(Float64, prob.u0)
    expected = n_nn + (pinn.param_estim ? length(md.ps) : 0)
    length(initial_θ) == expected || error(
        "OptimizationProblem u0 length $(length(initial_θ)) does not match expected $expected"
    )
    nparameters = length(initial_θ)
    ninv == (pinn.param_estim ? length(md.ps) : 0) || error(
        "`param` length $ninv does not match number of estimated PDE parameters $(length(md.ps))"
    )
    if ninv > 0
        initial_θ[(end - ninv + 1):end] .=
            Float64[Distributions.params(param[i])[1] for i in 1:ninv]
    end

    # Short MAP warmstart: the residual likelihood with small `phystd`/`bcstd` is
    # extremely peaked, so AdvancedHMC from a cold Lux init mixes poorly within the
    # sample budgets of the NeuralPDE 6 tests. Optimizing the System objective first
    # places the chain near the posterior mode without changing the log-density.
    if pretrain_iters > 0
        train_prob = remake(prob; u0 = initial_θ)
        tres = SciMLBase.solve(
            train_prob, OptimizationOptimisers.Adam(0.01); maxiters = pretrain_iters
        )
        θ_opt = hasproperty(tres, :original_sol) ? tres.original_sol.u : tres.u
        initial_θ = collect(Float64, θ_opt)
        if verbose
            obj = hasproperty(tres, :original_sol) ? tres.original_sol.objective : tres.objective
            @printf("Pretrain objective after %d Adam steps: %g\n", pretrain_iters, obj)
        end
    end

    stds = _block_stds(md.blocks, phystd, bcstd)
    residual_syms = Any[b.residual for b in md.blocks]

    priors = Distribution[
        MvNormal(
            priorsNNw[1] * ones(n_nn),
            Diagonal(abs2.(priorsNNw[2] .* ones(n_nn)))
        ),
    ]
    if ninv > 0
        append!(priors, param)
    end

    network_fns = map(nets) do net
        wrapper = getdefault(net.NN)
        return (X, θ) -> wrapper(X, θ)
    end

    ℓπ = PDELogTargetDensity(
        nparameters, prob, residual_syms, stds, priors, ninv, dataset, l2std,
        network_fns, net_lengths
    )

    @assert nchains ≥ 1 "number of chains must be greater than or equal to 1"

    Adaptor = Adaptorkwargs[:Adaptor]
    Metric = Adaptorkwargs[:Metric]
    targetacceptancerate = Adaptorkwargs[:targetacceptancerate]
    metric = Metric(nparameters)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    if verbose
        @printf("Current Physics Log-likelihood : %g\n", physics_loglikelihood(ℓπ, initial_θ))
        @printf("Current Prior Log-likelihood : %g\n", priorlogpdf(ℓπ, initial_θ))
        @printf(
            "Current SSE against dataset Log-likelihood : %g\n", L2LossData(ℓπ, initial_θ)
        )
    end

    function _run_chain(θ0)
        initial_ϵ = find_good_stepsize(hamiltonian, θ0)
        integrator = integratorchoice(Integratorkwargs, initial_ϵ)
        adaptor = adaptorchoice(
            Adaptor, MassMatrixAdaptor(metric),
            StepSizeAdaptor(targetacceptancerate, integrator)
        )
        kern = AdvancedHMC.make_kernel(Kernel, integrator)
        samples, stats = sample(
            hamiltonian, kern, θ0, draw_samples, adaptor;
            progress = progress, verbose = verbose
        )
        matrix_samples = hcat(samples...)
        mcmc_chain = MCMCChains.Chains(matrix_samples')
        fullsolution = BPINNstats(mcmc_chain, samples, stats)
        ensemblecurves, estimnnparams, estimated_params, timepoints = inference(
            samples, md, saveats, numensemble, ℓπ
        )
        return BPINNsolution(
            fullsolution, ensemblecurves, estimnnparams, estimated_params, timepoints
        )
    end

    if nchains != 1
        bpinnsols = Vector{Any}(undef, nchains)
        Threads.@threads for i in 1:nchains
            θ0 = vcat(randn(nparameters - ninv), initial_θ[(nparameters - ninv + 1):end])
            bpinnsols[i] = _run_chain(θ0)
        end
        return bpinnsols
    else
        sol = _run_chain(initial_θ)
        if verbose
            @printf("Sampling Complete.\n")
            @printf(
                "Final Physics Log-likelihood : %g\n",
                physics_loglikelihood(ℓπ, sol.original.samples[end])
            )
            @printf(
                "Final Prior Log-likelihood : %g\n",
                priorlogpdf(ℓπ, sol.original.samples[end])
            )
            @printf(
                "Final SSE against dataset Log-likelihood : %g\n",
                L2LossData(ℓπ, sol.original.samples[end])
            )
        end
        return sol
    end
end
