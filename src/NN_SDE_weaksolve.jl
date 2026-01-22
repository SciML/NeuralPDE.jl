
@concrete struct SDEPINN
    chain <: AbstractLuxLayer
    optimalg
    norm_loss_alg
    initial_parameters

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
    strategy <: Union{Nothing,AbstractTrainingStrategy}
    autodiff::Bool
    batch::Bool
    param_estim::Bool

    # For postprocessing - solution handling
    # xview::AbstractArray
    # tview::AbstractArray
    # phi::Phi

    dataset <: Union{Nothing,Vector,Vector{<:Vector}}
    additional_loss <: Union{Nothing,Function}
    kwargs
end

function SDEPINN(;
    chain,
    optimalg=nothing,
    norm_loss_alg=nothing,
    initial_parameters=nothing,
    x_0,
    x_end,
    Nt=50,
    dx=0.01,
    σ_var_bc=0.05,
    λ_ic=1.0,
    λ_norm=1.0,
    distrib=Normal(0.5, 0.01),
    strategy=nothing,
    autodiff=true,
    batch=false,
    param_estim=false,
    dataset=nothing,
    additional_loss=nothing,
    kwargs...
)
    return SDEPINN(
        chain,
        optimalg,
        norm_loss_alg,
        initial_parameters,
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

function SciMLBase.__solve(
    prob::SciMLBase.AbstractSDEProblem,
    alg::SDEPINN,
    args...;
    dt=nothing,
    abtol=1.0f-6,
    reltol=1.0f-3,
    saveat=nothing,
    tstops=nothing,
    maxiters=200,
    verbose=false,
    kwargs...,
)
    (; u0, tspan, f, g, p) = prob
    P = eltype(u0)
    t₀, t₁ = tspan

    absorbing_bc = false
    reflective_bc = true

    (; x_0, x_end, Nt, dx, σ_var_bc, λ_ic, λ_norm,
        distrib, optimalg, norm_loss_alg, initial_parameters, chain) = alg

    dt = (t₁ - t₀) / Nt
    ts = collect(t₀:dt:t₁)

    # Define FP PDE
    @parameters X, T
    @variables p̂(..)
    Dx = Differential(X)
    Dxx = Differential(X)^2
    Dt = Differential(T)

    J(x, T) = prob.f(x, p, T) * p̂(x, T) -
              P(0.5) * Dx((prob.g(x, p, T))^2 * p̂(x, T))

    # IC symbolic equation form
    f_icloss = if u0 isa Number
        (p̂(u0, t₀) - Distributions.pdf(distrib, u0) ~ P(0),)
    else
        (p̂(u0[i], t₀) .- Distributions.pdf(distrib[i], u0[i]) ~ P(0) for i in 1:length(u0))
    end

    # # inside optimization loss
    # # IC loss imposed pointwise only at t₀ and at x = u0, extremes of x domain.
    # ftest_icloss = if u0 isa Number
    #     (phi, θ) -> first(phi([u0, t₀], θ)) - Distributions.pdf(distrib, u0)
    # else
    #     (phi, θ) -> [phi([u0[i], t₀], θ) .-
    #                  Distributions.pdf(distrib[i], u0[i])
    #                  for i in 1:length(u0)]
    # end

    # function ic_loss(phi, θ)
    #     # println("inside ic : ", θ)
    #     #  insdie ic phi : 0.697219487224398
    #     # if integrated then optimization ends up focusing on mostly
    #     # non X₀ points (as they are more) and misses the peak properly.
    #     # (although a lognormal shape is still learnt)

    #     # I_ic = solve(IntegralProblem(f_ic, x_0, x_end, θ), norm_loss_alg,
    #     # HCubatureJL(),
    #     # reltol = 1e-8, abstol = 1e-8, maxiters = 10)[1]
    #     # return abs(I_ic) # I_ic loss AUC = 0?
    #     return sum(abs2, ftest_icloss(phi, θ)) # I_ic pointwise
    # end

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
        f_icloss...
    ]

    # absorbing Bcs
    if absorbing_bc
        @info "absorbing BCS used"

        bcs = vcat(bcs, [p̂(x_0, T) ~ P(0),
            p̂(x_end, T) ~ P(0)]...)
    end

    # reflecting Bcs 
    if reflective_bc
        @info "reflecting BCS used"

        bcs = vcat(bcs, [J(x_0, T) ~ P(0),
            J(x_end, T) ~ P(0)
        ]...)
    end

    domains = [X ∈ (x_0, x_end), T ∈ (t₀, t₁)]

    # Additional losses
    # Handle normloss and ICloss for vector NN outputs !! -> will need to adjst x0, x_end, u0 handling for this also !!

    σ_var_bc = 0.05 # must be narrow, dirac deltra function centering. (smaller this is, we drop NN from a taller point to learn)
    function norm_loss(phi, θ)
        loss = P(0)
        for t in ts
            # define integrand as a function of x only (t fixed)
            # perform ∫ f(x) dx over [x_0, x_end]
            phi_normloss(x, θ) = u0 isa Number ? first(phi([x, t], θ)) : phi([x, t], θ)
            I_est = solve(IntegralProblem(phi_normloss, x_0, x_end, θ), norm_loss_alg,
                reltol=1e-8, abstol=1e-8, maxiters=10)[1]
            loss += abs2(I_est - P(1))
        end
        return loss
    end

    function combined_additional(phi, θ, _)
        λ_norm * norm_loss(phi, θ)
    end

    # Discretization - GridTraining only
    discretization = PhysicsInformedNN(
        chain,
        GridTraining([dx, dt]);
        init_params=initial_parameters,
        additional_loss=combined_additional
    )

    @named pdesys = PDESystem(eq, bcs, domains, [X, T], [p̂(X, T)])
    opt_prob = discretize(pdesys, discretization)
    phi = discretization.phi

    sym = NeuralPDE.symbolic_discretize(pdesys, discretization)
    pde_losses = sym.loss_functions.pde_loss_functions
    bc_losses = sym.loss_functions.bc_loss_functions

    cb = function (p, l)
        (!verbose) && return false
        println("loss = ", l)
        println("pde = ", map(f -> f(p.u), pde_losses))
        println("bc  = ", map(f -> f(p.u), bc_losses))
        println("norm = ", norm_loss(phi, p.u))
        return false
    end

    res = Optimization.solve(
        opt_prob,
        optimalg;
        callback=cb,
        maxiters=maxiters,
        kwargs...
    )

    # postprocessing?
    return res, phi
end