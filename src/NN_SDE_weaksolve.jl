"""
    SDEPINN(;
        chain, x_0, x_end, optimalg = nothing, norm_loss_alg = nothing,
        initial_parameters = nothing, rng = Random.default_rng(),
        adtype = AutoZygote(), Nt = 20, dx = 0.05, σ_var_bc = 0.05,
        λ_ic = 1.0, λ_norm = 1.0, distrib = Normal(0.5, 0.01), strategy = nothing,
        autodiff = true, batch = false, param_estim = false, dataset = nothing,
        additional_loss = nothing, kwargs...
    )

Solve an `SDEProblem` by training a physics-informed neural network on its associated
Fokker-Planck equation over the spatial interval from `x_0` to `x_end`.

## Keyword Arguments

- `chain`: Neural network used to represent the probability density.
- `x_0`, `x_end`: Lower and upper endpoints of the spatial domain.
- `optimalg`: Optimizer used to train the network, or a vector/tuple of optimizers run
  in sequence; each stage warm-starts from the previous stage's parameters and is given
  `maxiters` iterations (`maxiters` may itself be a vector/tuple with one entry per stage).
- `norm_loss_alg`: Integration algorithm used by the normalization loss. The default
  applies adaptive Gauss-Kronrod quadrature on uniform spatial panels split further at
  up to nine quantiles of `distrib` and up to nine per positive-weight mixture component.
  The uniform panel width targets about ten standard deviations of a Gaussian with the
  same interquartile range as `distrib`, with limits of 1,024 uniform panels and 1,032
  panels in total after clamping and deduplicating knots. The automatic rule checks
  the integrated initial PDF mass against the enclosed CDF probability at relative
  tolerance `1e-4` and tightens quadrature tolerances if needed. If it cannot construct
  or validate the rule, the solve throws an `ArgumentError` with instructions to pass
  `norm_loss_alg` explicitly. An explicit algorithm bypasses automatic validation and
  can resolve narrow peaks that move away from the initial quantiles.
- `initial_parameters`: Initial network parameters. NeuralPDE draws them from `rng` when
  omitted.
- `rng`: Random number generator for network parameter initialization when
  `initial_parameters` is omitted. `SDEPINN` always uses `GridTraining`, so `rng`
  does not affect collocation points.
- `adtype`: AD backend of the training objective. Defaults to `AutoZygote()` because the
  normalization loss calls `Integrals.solve` on `norm_loss_alg`, which Enzyme cannot
  differentiate (`EnzymeRuntimeActivityError`).
- `Nt`: Number of temporal training points.
- `dx`: Spatial grid spacing used to evaluate the solution.
- `σ_var_bc`: Width of the Gaussian approximation to the initial condition.
- `λ_ic`, `λ_norm`: Initial-condition and normalization loss weights.
- `distrib`: Initial probability distribution.
- `strategy`: Optional training strategy.
- `autodiff`: Whether to use automatic differentiation for temporal derivatives.
- `batch`: Whether to batch training points.
- `param_estim`: Whether to estimate equation parameters with the network parameters.
- `dataset`: Optional observed data used during training.
- `additional_loss`: Optional function that contributes an additional training loss.
- `kwargs`: Additional keyword arguments forwarded to the Optimization.jl solve.

## Example

```julia
using Integrals, Lux, NeuralPDE, OptimizationOptimJL

chain = Chain(Dense(2, 16, tanh), Dense(16, 1))
alg = SDEPINN(
    chain = chain,
    optimalg = BFGS(),
    norm_loss_alg = HCubatureJL(),
    x_0 = -4.0,
    x_end = 4.0,
)
```
"""
@concrete struct SDEPINN <: SciMLBase.AbstractSDEAlgorithm
    chain <: AbstractLuxLayer
    optimalg
    norm_loss_alg
    initial_parameters
    rng <: AbstractRNG
    adtype

    # domain + discretization
    x_0::Float64
    x_end::Float64
    Nt::Int
    dx::Float64

    # IC & normalization
    σ_var_bc::Float64
    λ_ic::Float64
    λ_norm::Float64
    distrib::Distributions.Distribution

    # solver options
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    autodiff::Bool
    batch::Bool
    param_estim::Bool

    dataset <: Union{Nothing, Vector, Vector{<:Vector}}
    additional_loss <: Union{Nothing, Function}
    kwargs
end

function SDEPINN(;
        chain,
        optimalg = nothing,
        norm_loss_alg = nothing,
        initial_parameters = nothing,
        rng::AbstractRNG = Random.default_rng(),
        adtype = AutoZygote(),
        x_0,
        x_end,
        Nt = 20,
        dx = 0.05,
        σ_var_bc = 0.05,
        λ_ic = 1.0,
        λ_norm = 1.0,
        distrib = Normal(0.5, 0.01),
        strategy = nothing,
        autodiff = true,
        batch = false,
        param_estim = false,
        dataset = nothing,
        additional_loss = nothing,
        kwargs...
    )
    return SDEPINN(
        chain,
        optimalg,
        norm_loss_alg,
        initial_parameters,
        rng,
        adtype,
        x_0,
        x_end,
        Nt,
        dx,
        σ_var_bc,
        λ_ic,
        λ_norm,
        distrib,
        strategy,
        autodiff,
        batch,
        param_estim,
        dataset,
        additional_loss,
        kwargs
    )
end

struct _NormLossAlg{A, T}
    alg::A
    breakpoints::Vector{T}
    reltol::Float64
    abstol::Float64
end

"""
    resolve_norm_loss_alg(norm_loss_alg, x_0, x_end, distrib)

Prepare and validate the automatic normalization quadrature before automatic
differentiation, then reuse it on each time slice.
"""
function resolve_norm_loss_alg(norm_loss_alg, x_0, x_end, distrib)
    norm_loss_alg === nothing || return norm_loss_alg
    mass_tolerance = 1.0e-4
    tail_probability = mass_tolerance / 10_000
    probabilities = (
        tail_probability, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
        0.875, 1 - tail_probability,
    )

    invalid_rule(reason) = ArgumentError(
        "could not select the default normalization quadrature for initial " *
            "distribution $(repr(distrib)): $reason; pass `norm_loss_alg` explicitly"
    )
    function quantiles_for(d)
        quantiles = try
            map(p -> Distributions.quantile(d, p), probabilities)
        catch err
            err isa InterruptException && rethrow()
            throw(invalid_rule("could not compute quantiles of $(repr(d)) ($(sprint(showerror, err)))"))
        end
        all(isfinite, quantiles) || throw(invalid_rule("nonfinite quantiles of $(repr(d))"))
        allunique(quantiles) || throw(
            invalid_rule("could not resolve the quantiles of $(repr(d)) at floating-point precision")
        )
        return quantiles
    end

    quantiles = quantiles_for(distrib)
    knots = [x_0, x_end, quantiles...]
    if distrib isa Distributions.AbstractMixtureModel
        for (component, weight) in zip(Distributions.components(distrib), Distributions.probs(distrib))
            weight > 0 || continue
            append!(knots, quantiles_for(component))
        end
    end

    breakpoints = Set(clamp.(knots, x_0, x_end))
    max_panels = 1032
    uniform_limit = min(1024, max_panels - (length(breakpoints) - 2))
    uniform_limit >= 1 || throw(invalid_rule("quantile knots exceed the $max_panels-panel limit"))
    interquartile_range = quantiles[7] - quantiles[3]
    gaussian_iqr = 2 * Distributions.quantile(Distributions.Normal(), 0.75)
    panel_width = 10 * (interquartile_range / gaussian_iqr)
    panel_count = min(ceil(Int, clamp((x_end - x_0) / panel_width, 4, 1024)), uniform_limit)
    union!(breakpoints, range(x_0, x_end; length = panel_count + 1))
    sorted_breakpoints = sort!(collect(breakpoints))

    expected_mass = try
        Distributions.cdf(distrib, x_end) - Distributions.cdf(distrib, x_0)
    catch err
        err isa InterruptException && rethrow()
        throw(invalid_rule("could not compute enclosed CDF probability ($(sprint(showerror, err)))"))
    end
    initial_pdf(x, d) = Distributions.pdf(d, x)
    alg = QuadGKJL()
    for tolerance in (mass_tolerance, mass_tolerance / 10, mass_tolerance / 100)
        rule = _NormLossAlg(alg, sorted_breakpoints, tolerance, tolerance)
        mass = try
            normalization_mass(initial_pdf, x_0, x_end, distrib, rule)
        catch err
            err isa InterruptException && rethrow()
            throw(invalid_rule("could not integrate the initial PDF ($(sprint(showerror, err)))"))
        end
        isapprox(mass, expected_mass; rtol = mass_tolerance) && return rule
    end
    throw(
        invalid_rule(
            "integrated initial PDF mass does not match the enclosed CDF probability " *
                "$expected_mass within relative tolerance $mass_tolerance"
        )
    )
end

"""
    normalization_mass(integrand, x_0, x_end, θ, norm_loss_alg)

Integrate `integrand` over `[x_0, x_end]` for the normalization loss.
"""
function normalization_mass(integrand, x_0, x_end, θ, norm_loss_alg)
    if norm_loss_alg isa _NormLossAlg
        (; alg, breakpoints, reltol, abstol) = norm_loss_alg
        panel_count = length(breakpoints) - 1
        panel_abstol = abstol / panel_count
        masses = map(1:panel_count) do i
            prob = IntegralProblem(integrand, (breakpoints[i], breakpoints[i + 1]), θ)
            solve(prob, alg; reltol, abstol = panel_abstol)[1]
        end
        return sum(masses)
    end
    prob = IntegralProblem(integrand, (x_0, x_end), θ)
    return solve(prob, norm_loss_alg; reltol = 1.0e-4, abstol = 1.0e-4)[1]
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSDEProblem,
        alg::SDEPINN,
        args...;
        dt = nothing,
        abtol = 1.0f-6,
        reltol = 1.0f-3,
        saveat = nothing,
        tstops = nothing,
        maxiters = 200,
        verbose = false,
        kwargs...,
    )
    (; u0, tspan, f, g, p) = prob
    P = eltype(u0)
    t₀, t₁ = tspan

    absorbing_bc = false
    reflective_bc = true

    (;
        x_0, x_end, Nt, dx, σ_var_bc, λ_ic, λ_norm,
        distrib, optimalg, norm_loss_alg, initial_parameters, rng, adtype, chain,
    ) = alg

    norm_loss_alg = resolve_norm_loss_alg(norm_loss_alg, x_0, x_end, distrib)

    dt = (t₁ - t₀) / Nt
    ts = collect(t₀:dt:t₁)

    # Define FP PDE
    @parameters X, T
    @variables p̂(..)
    Dx = Differential(X)
    Dxx = Differential(X)^2
    Dt = Differential(T)

    # use product rule to avoid Dx(0) in symbolic equations for BC.
    J(X, T) = prob.f(X, p, T) * p̂(X, T) -
        P(0.5) * (
        ((prob.g(X, p, T))^2 * Dx(p̂(X, T))) +
            (p̂(X, T) * Dx((prob.g(X, p, T))^2))
    )

    # IC symbolic equation form
    f_icloss = if u0 isa Number
        (p̂(u0, t₀) - Distributions.pdf(distrib, u0) ~ P(0),)
    else
        (p̂(u0[i], t₀) .- Distributions.pdf(distrib[i], u0[i]) ~ P(0) for i in 1:length(u0))
    end

    eq = Dt(p̂(X, T)) ~ -Dx(f(X, p, T) * p̂(X, T)) +
        P(0.5) * Dxx((g(X, p, T))^2 * p̂(X, T))

    # if we try to use p=0 and normalization it works
    # however if we increase the x domainby  too much on any side:
    # The Normalization PDF mass although "conserved" inside domain
    # can be forced to spread in different regions.

    bcs = [
        # No probability enters or leaves the domain
        # Total mass is conserved
        # Matches an SDE on a truncated but reflecting domain BC

        # IC LOSS (it's getting amplified by the number of training points.)
        f_icloss...,
    ]

    # absorbing Bcs
    if absorbing_bc
        @info "absorbing BCS used"

        bcs = vcat(
            bcs, [
                p̂(x_0, T) ~ P(0),
                p̂(x_end, T) ~ P(0),
            ]...
        )
    end

    # reflecting Bcs
    if reflective_bc
        @info "reflecting BCS used"

        bcs = vcat(
            bcs, [
                J(x_0, T) ~ P(0),
                J(x_end, T) ~ P(0),
            ]...
        )
    end

    domains = [X ∈ (x_0, x_end), T ∈ (t₀, t₁)]

    # Additional losses
    # TODO Handle normloss and ICloss for vector NN outputs !! -> will need to adjst x0, x_end, u0 handling for this

    σ_var_bc = 0.05 # must be narrow, dirac deltra function centering. (smaller this is, we drop NN from a taller point to learn)
    function norm_loss(phi, θ)
        loss = P(0)
        for t in ts
            # define integrand as a function of x only (t fixed)
            # perform ∫ f(x) dx over [x_0, x_end]
            phi_normloss(x, θ) = u0 isa Number ? first(phi([x, t], θ)) : phi([x, t], θ)
            I_est = normalization_mass(phi_normloss, x_0, x_end, θ, norm_loss_alg)
            loss += abs2(I_est - P(1))
        end
        return loss
    end

    function combined_additional(phi, θ, _)
        return λ_norm * norm_loss(phi.p̂, θ.p̂)
    end

    # Discretization - GridTraining only
    discretization = PhysicsInformedNN(
        chain,
        GridTraining([dx, dt]);
        init_params = initial_parameters,
        rng = rng,
        additional_loss = combined_additional
    )

    @named pdesys = PDESystem(eq, bcs, domains, [X, T], [p̂(X, T)])
    opt_prob = discretize(pdesys, discretization; adtype = adtype)
    md = pinn_metadata(opt_prob)
    net = only(md.networks)
    wrapper = getdefault(net.NN)
    phi = (x, θ) -> wrapper(x, θ)

    cb = function (p, l)
        # DiffEqBase now passes DEVerbosity by default; keep these debug prints opt-in.
        verbose === true || return false
        println("loss = ", l)
        println("norm = ", norm_loss(phi, p.u))
        return false
    end

    opts, iters = _sdepinn_stages(optimalg, maxiters)
    res = nothing
    for (opt, mi) in zip(opts, iters)
        res = Optimization.solve(
            opt_prob, opt; callback = cb, maxiters = mi, kwargs...
        )
        opt_prob = remake(opt_prob; u0 = res.original_sol.u)
    end

    # The PDE solution wrapper is unwrapped so that `res.u` stays the parameter vector.
    return res.original_sol, phi
end

"""
    _sdepinn_stages(optimalg, maxiters)

Normalize `optimalg` / `maxiters` into equal-length stage sequences. A scalar
`maxiters` is broadcast to every stage; a vector/tuple must match `optimalg`.
"""
function _sdepinn_stages(optimalg, maxiters)
    opts = optimalg isa Union{Tuple, AbstractVector} ? optimalg : (optimalg,)
    iters = if maxiters isa Union{Tuple, AbstractVector}
        length(maxiters) == length(opts) || throw(
            ArgumentError(
                "`maxiters` has $(length(maxiters)) entries but `optimalg` has \
                $(length(opts)) stages."
            )
        )
        maxiters
    else
        ntuple(_ -> maxiters, length(opts))
    end
    return opts, iters
end
