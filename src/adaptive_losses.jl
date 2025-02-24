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

function NonAdaptiveLoss{T}(; pde_loss_weights = 1.0, bc_loss_weights = 1.0,
        additional_loss_weights = 1.0) where {T <: Real}
    return NonAdaptiveLoss{T}(
        vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
        vectorify(additional_loss_weights, T))
end

NonAdaptiveLoss(; kwargs...) = NonAdaptiveLoss{Float64}(; kwargs...)

@closure function generate_adaptive_loss_function(::PINNRepresentation, ::NonAdaptiveLoss, _, __)
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

function GradientScaleAdaptiveLoss{T}(reweight_every::Int;
        weight_change_inertia = 0.9, pde_loss_weights = 1.0,
        bc_loss_weights = 1.0, additional_loss_weights = 1.0) where {T <: Real}
    return GradientScaleAdaptiveLoss{T}(reweight_every, weight_change_inertia,
        vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
        vectorify(additional_loss_weights, T))
end

function GradientScaleAdaptiveLoss(args...; kwargs...)
    return GradientScaleAdaptiveLoss{Float64}(args...; kwargs...)
end

@closure function generate_adaptive_loss_function(pinnrep::PINNRepresentation,
        adaloss::GradientScaleAdaptiveLoss, pde_loss_functions, bc_loss_functions)
    weight_change_inertia = adaloss.weight_change_inertia
    iteration = pinnrep.iteration
    adaloss_T = eltype(adaloss.pde_loss_weights)

    return (θ, pde_losses, bc_losses) -> begin
        if iteration[] % adaloss.reweight_every == 0
            # the paper assumes a single pde loss function, so here we grab the maximum of
            # the maximums of each pde loss function
            pde_grads_maxes = [maximum(abs, only(Zygote.gradient(pde_loss_function, θ)))
                               for pde_loss_function in pde_loss_functions]
            pde_grads_max = maximum(pde_grads_maxes)
            bc_grads_mean = [mean(abs, only(Zygote.gradient(bc_loss_function, θ)))
                             for bc_loss_function in bc_loss_functions]

            nonzero_divisor_eps = adaloss_T isa Float64 ? 1e-11 : convert(adaloss_T, 1e-7)
            bc_loss_weights_proposed = pde_grads_max ./
                                       (bc_grads_mean .+ nonzero_divisor_eps)
            adaloss.bc_loss_weights .= weight_change_inertia .*
                                       adaloss.bc_loss_weights .+
                                       (1 .- weight_change_inertia) .*
                                       bc_loss_weights_proposed
            logscalar(pinnrep.logger, pde_grads_max, "adaptive_loss/pde_grad_max",
                iteration[])
            logvector(pinnrep.logger, pde_grads_maxes, "adaptive_loss/pde_grad_maxes",
                iteration[])
            logvector(pinnrep.logger, bc_grads_mean, "adaptive_loss/bc_grad_mean",
                iteration[])
            logvector(pinnrep.logger, adaloss.bc_loss_weights,
                "adaptive_loss/bc_loss_weights", iteration[])
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

function MiniMaxAdaptiveLoss{T}(reweight_every::Int; pde_max_optimiser = Adam(1e-4),
        bc_max_optimiser = Adam(0.5), pde_loss_weights = 1.0, bc_loss_weights = 1.0,
        additional_loss_weights = 1.0) where {T <: Real}
    return MiniMaxAdaptiveLoss{T}(reweight_every, pde_max_optimiser, bc_max_optimiser,
        vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
        vectorify(additional_loss_weights, T))
end

MiniMaxAdaptiveLoss(args...; kwargs...) = MiniMaxAdaptiveLoss{Float64}(args...; kwargs...)

@closure function generate_adaptive_loss_function(pinnrep::PINNRepresentation,
        adaloss::MiniMaxAdaptiveLoss, _, __)
    pde_max_optimiser_setup = Optimisers.setup(
        adaloss.pde_max_optimiser, adaloss.pde_loss_weights)
    bc_max_optimiser_setup = Optimisers.setup(
        adaloss.bc_max_optimiser, adaloss.bc_loss_weights)
    iteration = pinnrep.iteration

    return (θ, pde_losses, bc_losses) -> begin
        if iteration[] % adaloss.reweight_every == 0
            Optimisers.update!(
                pde_max_optimiser_setup, adaloss.pde_loss_weights, -pde_losses)
            Optimisers.update!(bc_max_optimiser_setup, adaloss.bc_loss_weights, -bc_losses)
            logvector(pinnrep.logger, adaloss.pde_loss_weights,
                "adaptive_loss/pde_loss_weights", iteration[])
            logvector(pinnrep.logger, adaloss.bc_loss_weights,
                "adaptive_loss/bc_loss_weights", iteration[])
        end
        return nothing
    end
end
