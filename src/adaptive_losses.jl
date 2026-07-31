abstract type AbstractAdaptiveLoss end

# Utils
vectorify(x::Vector, ::Type{T}) where {T <: Real} = T.(x)
vectorify(x, ::Type{T}) where {T <: Real} = T[convert(T, x)]

# Dispatches
"""
    NonAdaptiveLoss(; pde_loss_weights = 1.0,
                      bc_loss_weights = 1.0,
                      additional_loss_weights = 1.0)

A way of loss weighting the components of the loss function in the total sum that does not
change during optimization
"""
@concrete mutable struct NonAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
end

function NonAdaptiveLoss{T}(;
        pde_loss_weights = 1.0, bc_loss_weights = 1.0,
        additional_loss_weights = 1.0
    ) where {T <: Real}
    return NonAdaptiveLoss{T}(
        vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
        vectorify(additional_loss_weights, T)
    )
end

NonAdaptiveLoss(; kwargs...) = NonAdaptiveLoss{Float64}(; kwargs...)

function generate_adaptive_loss_function(::PINNRepresentation, ::NonAdaptiveLoss, _, __)
    return Returns(nothing)
end

"""
    GradientScaleAdaptiveLoss(reweight_every;
                              weight_change_inertia = 0.9,
                              pde_loss_weights = 1.0,
                              bc_loss_weights = 1.0,
                              additional_loss_weights = 1.0)

A way of adaptively reweighting the components of the loss function in the total sum such
that BC_i loss weights are scaled by the exponential moving average of
max(|∇pde_loss|) / mean(|∇bc_i_loss|)).

## Positional Arguments

* `reweight_every`: how often to reweight the BC loss functions, measured in iterations.
  Reweighting is somewhat expensive since it involves evaluating the gradient of each
  component loss function,

## Keyword Arguments

* `weight_change_inertia`: a real number that represents the inertia of the exponential
  moving average of the BC weight changes,

## References

Understanding and mitigating gradient pathologies in physics-informed neural networks
Sifan Wang, Yujun Teng, Paris Perdikaris
https://arxiv.org/abs/2001.04536v1

With code reference:
https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs
"""
@concrete mutable struct GradientScaleAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    reweight_every::Int
    weight_change_inertia::T
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
end

function GradientScaleAdaptiveLoss{T}(
        reweight_every::Int;
        weight_change_inertia = 0.9, pde_loss_weights = 1.0,
        bc_loss_weights = 1.0, additional_loss_weights = 1.0
    ) where {T <: Real}
    return GradientScaleAdaptiveLoss{T}(
        reweight_every, weight_change_inertia,
        vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
        vectorify(additional_loss_weights, T)
    )
end

function GradientScaleAdaptiveLoss(args...; kwargs...)
    return GradientScaleAdaptiveLoss{Float64}(args...; kwargs...)
end

function generate_adaptive_loss_function(
        pinnrep::PINNRepresentation,
        adaloss::GradientScaleAdaptiveLoss, pde_loss_functions, bc_loss_functions
    )
    weight_change_inertia = adaloss.weight_change_inertia
    iteration = pinnrep.iteration
    adaloss_T = eltype(adaloss.pde_loss_weights)

    return (
        θ,
        pde_losses,
        bc_losses,
    ) -> begin
        if iteration[] % adaloss.reweight_every == 0
            # the paper assumes a single pde loss function, so here we grab the maximum of
            # the maximums of each pde loss function
            pde_grads_maxes = [
                maximum(abs, only(Zygote.gradient(pde_loss_function, θ)))
                    for pde_loss_function in pde_loss_functions
            ]
            pde_grads_max = maximum(pde_grads_maxes)
            bc_grads_mean = [
                mean(abs, only(Zygote.gradient(bc_loss_function, θ)))
                    for bc_loss_function in bc_loss_functions
            ]

            nonzero_divisor_eps = adaloss_T isa Float64 ? 1.0e-11 : convert(adaloss_T, 1.0e-7)
            bc_loss_weights_proposed = pde_grads_max ./
                (bc_grads_mean .+ nonzero_divisor_eps)
            adaloss.bc_loss_weights .= weight_change_inertia .*
                adaloss.bc_loss_weights .+
                (1 .- weight_change_inertia) .*
                bc_loss_weights_proposed
            logscalar(
                pinnrep.logger, pde_grads_max, "adaptive_loss/pde_grad_max",
                iteration[]
            )
            logvector(
                pinnrep.logger, pde_grads_maxes, "adaptive_loss/pde_grad_maxes",
                iteration[]
            )
            logvector(
                pinnrep.logger, bc_grads_mean, "adaptive_loss/bc_grad_mean",
                iteration[]
            )
            logvector(
                pinnrep.logger, adaloss.bc_loss_weights,
                "adaptive_loss/bc_loss_weights", iteration[]
            )
        end
        return nothing
    end
end

"""
    MiniMaxAdaptiveLoss(reweight_every;
                        pde_max_optimiser = OptimizationOptimisers.Adam(1e-4),
                        bc_max_optimiser = OptimizationOptimisers.Adam(0.5),
                        pde_loss_weights = 1, bc_loss_weights = 1,
                        additional_loss_weights = 1)

A way of adaptively reweighting the components of the loss function in the total sum such
that the loss weights are maximized by an internal optimizer, which leads to a behavior
where loss functions that have not been satisfied get a greater weight.

## Positional Arguments

* `reweight_every`: how often to reweight the PDE and BC loss functions, measured in
  iterations.  Reweighting is cheap since it re-uses the value of loss functions generated
  during the main optimization loop.

## Keyword Arguments

* `pde_max_optimiser`: a OptimizationOptimisers optimiser that is used internally to
  maximize the weights of the PDE loss functions.
* `bc_max_optimiser`: a OptimizationOptimisers optimiser that is used internally to maximize
  the weights of the BC loss functions.

## References

Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism
Levi McClenny, Ulisses Braga-Neto
https://arxiv.org/abs/2009.04544
"""
@concrete mutable struct MiniMaxAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    reweight_every::Int
    pde_max_optimiser <: Optimisers.AbstractRule
    bc_max_optimiser <: Optimisers.AbstractRule
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
end

function MiniMaxAdaptiveLoss{T}(
        reweight_every::Int; pde_max_optimiser = Adam(1.0e-4),
        bc_max_optimiser = Adam(0.5), pde_loss_weights = 1.0, bc_loss_weights = 1.0,
        additional_loss_weights = 1.0
    ) where {T <: Real}
    return MiniMaxAdaptiveLoss{T}(
        reweight_every, pde_max_optimiser, bc_max_optimiser,
        vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
        vectorify(additional_loss_weights, T)
    )
end

MiniMaxAdaptiveLoss(args...; kwargs...) = MiniMaxAdaptiveLoss{Float64}(args...; kwargs...)

function generate_adaptive_loss_function(
        pinnrep::PINNRepresentation,
        adaloss::MiniMaxAdaptiveLoss, _, __
    )
    pde_max_optimiser_setup = Optimisers.setup(
        adaloss.pde_max_optimiser, adaloss.pde_loss_weights
    )
    bc_max_optimiser_setup = Optimisers.setup(
        adaloss.bc_max_optimiser, adaloss.bc_loss_weights
    )
    iteration = pinnrep.iteration

    return (
        θ,
        pde_losses,
        bc_losses,
    ) -> begin
        if iteration[] % adaloss.reweight_every == 0
            Optimisers.update!(
                pde_max_optimiser_setup, adaloss.pde_loss_weights, -pde_losses
            )
            Optimisers.update!(bc_max_optimiser_setup, adaloss.bc_loss_weights, -bc_losses)
            logvector(
                pinnrep.logger, adaloss.pde_loss_weights,
                "adaptive_loss/pde_loss_weights", iteration[]
            )
            logvector(
                pinnrep.logger, adaloss.bc_loss_weights,
                "adaptive_loss/bc_loss_weights", iteration[]
            )
        end
        return nothing
    end
end

# ── Softmax helper (no external dependency) ───────────────────────────────────
function _softmax(x::AbstractVector{T}) where {T}
    e = exp.(x .- maximum(x))   # subtract max for numerical stability
    return e ./ sum(e)
end

"""
    SoftAdaptAdaptiveLoss(reweight_every;
                          α = 0.1,
                          pde_loss_weights = 1.0,
                          bc_loss_weights = 1.0,
                          additional_loss_weights = 1.0)

An adaptive loss weighting strategy based on the relative rate of change of each
loss component. Weights are assigned proportionally via a softmax over the
normalised loss rates, so components that are growing faster receive a larger weight.

No gradient computations are required; reweighting cost is O(N) in the number of
loss terms, making it cheaper than `GradientScaleAdaptiveLoss`.

## Positional Arguments

* `reweight_every`: how often (in iterations) to update the loss weights.

## Keyword Arguments

* `α`: temperature parameter controlling the sharpness of the softmax
  (default `0.1`). Higher values make the weighting more aggressive.

## Algorithm

```
rate_i(t) = (L_i(t) - L_i(t-1)) / (L_i(t-1) + ε)
λ_i(t)    = softmax(α · rate(t)) × N
```

## References

Heydari, A. A., Thompson, C. A., & Mehmood, A. (2019).
SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks with
Multi-Part Loss Functions. arXiv:1912.12355.
https://arxiv.org/abs/1912.12355
"""
@concrete mutable struct SoftAdaptAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    reweight_every::Int
    α::T
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
    # stored across reweight steps
    prev_pde_losses::Vector{T}
    prev_bc_losses::Vector{T}
end

function SoftAdaptAdaptiveLoss{T}(
        reweight_every::Int;
        α = T(0.1),
        pde_loss_weights = 1.0,
        bc_loss_weights = 1.0,
        additional_loss_weights = 1.0
    ) where {T <: Real}
    pde_w = vectorify(pde_loss_weights, T)
    bc_w = vectorify(bc_loss_weights, T)
    return SoftAdaptAdaptiveLoss{T}(
        reweight_every, T(α),
        pde_w, bc_w, vectorify(additional_loss_weights, T),
        zeros(T, length(pde_w)), zeros(T, length(bc_w))
    )
end

SoftAdaptAdaptiveLoss(args...; kwargs...) = SoftAdaptAdaptiveLoss{Float64}(args...; kwargs...)

function generate_adaptive_loss_function(
        pinnrep::PINNRepresentation,
        adaloss::SoftAdaptAdaptiveLoss, _, __
    )
    iteration = pinnrep.iteration
    T = eltype(adaloss.pde_loss_weights)
    ε = T(1.0e-8)
    initialized = Ref(false)

    return (θ, pde_losses, bc_losses) -> begin
        # Seed previous losses on the very first call
        if !initialized[]
            adaloss.prev_pde_losses .= pde_losses
            adaloss.prev_bc_losses .= bc_losses
            initialized[] = true
        end

        if iteration[] % adaloss.reweight_every == 0
            # Resize stored vectors if the loss dimension changed (e.g. first real call)
            if length(adaloss.prev_pde_losses) != length(pde_losses)
                adaloss.prev_pde_losses = T.(pde_losses)
            end
            if length(adaloss.prev_bc_losses) != length(bc_losses)
                adaloss.prev_bc_losses = T.(bc_losses)
            end

            all_losses = vcat(T.(pde_losses), T.(bc_losses))
            all_prev = vcat(adaloss.prev_pde_losses, adaloss.prev_bc_losses)
            N = length(all_losses)

            rates = (all_losses .- all_prev) ./ (all_prev .+ ε)
            weights = _softmax(adaloss.α .* rates) .* T(N)

            n_pde = length(pde_losses)
            adaloss.pde_loss_weights .= weights[1:n_pde]
            adaloss.bc_loss_weights .= weights[(n_pde + 1):end]

            adaloss.prev_pde_losses .= T.(pde_losses)
            adaloss.prev_bc_losses .= T.(bc_losses)

            logvector(
                pinnrep.logger, adaloss.pde_loss_weights,
                "adaptive_loss/pde_loss_weights", iteration[]
            )
            logvector(
                pinnrep.logger, adaloss.bc_loss_weights,
                "adaptive_loss/bc_loss_weights", iteration[]
            )
        end
        return nothing
    end
end

"""
    ReLoBRaLoAdaptiveLoss(reweight_every;
                          α = 1.0,
                          β = 0.9,
                          pde_loss_weights = 1.0,
                          bc_loss_weights = 1.0,
                          additional_loss_weights = 1.0)

Relative Loss Balancing with Random Lookback (ReLoBRaLo). Adaptively reweights
loss components by comparing their current value to a randomly chosen past
reference — either the initial loss or the most recent checkpoint — controlled
by the lookback probability `β`.

This makes the method more robust to short-term loss oscillations than
purely incremental strategies like `SoftAdaptAdaptiveLoss`. No gradient
computations are required.

## Positional Arguments

* `reweight_every`: how often (in iterations) to update the loss weights.

## Keyword Arguments

* `α`: temperature parameter for the softmax (default `1.0`).
* `β`: probability of using the *previous* checkpoint as reference instead
  of the *initial* losses (default `0.9`). Setting `β = 0` always uses the
  initial losses; `β = 1` always uses the most recent checkpoint.

## Algorithm

```
ρ    ~ Bernoulli(β)
t₀   = ρ · t_prev + (1 - ρ) · t_init
λ_i  = softmax(α · L_i(t) / (L_i(t₀) + ε)) × N
```

## References

Bischof, R., & Kraus, M. (2021).
Multi-Objective Loss Balancing for Physics-Informed Deep Learning.
arXiv:2110.09813. https://arxiv.org/abs/2110.09813
"""
@concrete mutable struct ReLoBRaLoAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    reweight_every::Int
    α::T
    β::T
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
    # stored across reweight steps
    init_pde_losses::Vector{T}
    init_bc_losses::Vector{T}
    prev_pde_losses::Vector{T}
    prev_bc_losses::Vector{T}
end

function ReLoBRaLoAdaptiveLoss{T}(
        reweight_every::Int;
        α = T(1.0),
        β = T(0.9),
        pde_loss_weights = 1.0,
        bc_loss_weights = 1.0,
        additional_loss_weights = 1.0
    ) where {T <: Real}
    pde_w = vectorify(pde_loss_weights, T)
    bc_w = vectorify(bc_loss_weights, T)
    return ReLoBRaLoAdaptiveLoss{T}(
        reweight_every, T(α), T(β),
        pde_w, bc_w, vectorify(additional_loss_weights, T),
        zeros(T, length(pde_w)), zeros(T, length(bc_w)),  # init (filled on first call)
        zeros(T, length(pde_w)), zeros(T, length(bc_w))   # prev
    )
end

ReLoBRaLoAdaptiveLoss(args...; kwargs...) = ReLoBRaLoAdaptiveLoss{Float64}(args...; kwargs...)

function generate_adaptive_loss_function(
        pinnrep::PINNRepresentation,
        adaloss::ReLoBRaLoAdaptiveLoss, _, __
    )
    iteration = pinnrep.iteration
    T = eltype(adaloss.pde_loss_weights)
    ε = T(1.0e-8)
    initialized = Ref(false)

    return (θ, pde_losses, bc_losses) -> begin
        # Record initial losses on the very first call
        if !initialized[]
            adaloss.init_pde_losses .= T.(pde_losses)
            adaloss.init_bc_losses .= T.(bc_losses)
            adaloss.prev_pde_losses .= T.(pde_losses)
            adaloss.prev_bc_losses .= T.(bc_losses)
            initialized[] = true
        end

        if iteration[] % adaloss.reweight_every == 0
            use_prev = rand() < adaloss.β
            ref_pde = use_prev ? adaloss.prev_pde_losses : adaloss.init_pde_losses
            ref_bc = use_prev ? adaloss.prev_bc_losses : adaloss.init_bc_losses

            all_losses = vcat(T.(pde_losses), T.(bc_losses))
            all_ref = vcat(ref_pde, ref_bc)
            N = length(all_losses)

            ratios = all_losses ./ (all_ref .+ ε)
            weights = _softmax(adaloss.α .* ratios) .* T(N)

            n_pde = length(pde_losses)
            adaloss.pde_loss_weights .= weights[1:n_pde]
            adaloss.bc_loss_weights .= weights[(n_pde + 1):end]

            adaloss.prev_pde_losses .= T.(pde_losses)
            adaloss.prev_bc_losses .= T.(bc_losses)

            logvector(
                pinnrep.logger, adaloss.pde_loss_weights,
                "adaptive_loss/pde_loss_weights", iteration[]
            )
            logvector(
                pinnrep.logger, adaloss.bc_loss_weights,
                "adaptive_loss/bc_loss_weights", iteration[]
            )
        end
        return nothing
    end
end
