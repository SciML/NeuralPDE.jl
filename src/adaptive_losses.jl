abstract type AbstractAdaptiveLoss end

"""
A way of weighting the components of the loss function in the total sum that does not change during optimization

* `pde_loss_weights`: either a scalar (which will be broadcast) or vector the size of the number of PDE equations, which describes the weight the respective PDE loss has in the full loss sum,
* `bc_loss_weights`: either a scalar (which will be broadcast) or vector the size of the number of BC equations, which describes the weight the respective BC loss has in the full loss sum,
* `additional_loss_weights`: a scalar which describes the weight the additional loss function has in the full loss sum,
"""
mutable struct NonAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T} 
    SciMLBase.@add_kwonly function NonAdaptiveLoss{T}(;pde_loss_weights=1, bc_loss_weights=1, additional_loss_weights=1) where T <: Real
        new(vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T), vectorify(additional_loss_weights, T))
    end
end

# default to Float64
SciMLBase.@add_kwonly function NonAdaptiveLoss(;pde_loss_weights=1, bc_loss_weights=1, additional_loss_weights=1) 
    NonAdaptiveLoss{Float64}(;pde_loss_weights=pde_loss_weights, bc_loss_weights=bc_loss_weights, additional_loss_weights=additional_loss_weights)
end

"""
A way of adaptively reweighting the components of the loss function in the total sum such that BC_i loss weights are scaled by the exponential moving average of max(|∇pde_loss|)/mean(|∇bc_i_loss|) )

* `reweight_every`: how often to reweight the BC loss functions, measured in iterations.  reweighting is somewhat expensive since it involves evaluating the gradient of each component loss function,
* `weight_change_inertia`: a real number that represents the inertia of the exponential moving average of the BC weight changes,
* `pde_loss_weights`: either a scalar (which will be broadcast) or vector the size of the number of PDE equations, which describes the weight the respective PDE loss has in the full loss sum,
* `bc_loss_weights`: either a scalar (which will be broadcast) or vector the size of the number of BC equations, which describes the initial weight the respective BC loss has in the full loss sum,
* `additional_loss_weights`: a scalar which describes the weight the additional loss function has in the full loss sum, this is currently not adaptive and will be constant with this adaptive loss,

from paper
Understanding and mitigating gradient pathologies in physics-informed neural networks 
Sifan Wang, Yujun Teng, Paris Perdikaris
https://arxiv.org/abs/2001.04536v1
with code reference
https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs
"""
mutable struct GradientScaleAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    reweight_every::Int64
    weight_change_inertia::T
    pde_loss_weights::Vector{T} 
    bc_loss_weights::Vector{T} 
    additional_loss_weights::Vector{T} 
    SciMLBase.@add_kwonly function GradientScaleAdaptiveLoss{T}(reweight_every; weight_change_inertia=0.9, pde_loss_weights=1, bc_loss_weights=1, additional_loss_weights=1) where T <: Real
        new(convert(Int64, reweight_every), convert(T, weight_change_inertia), vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T), vectorify(additional_loss_weights, T))
    end
end
# default to Float64
SciMLBase.@add_kwonly function GradientScaleAdaptiveLoss(reweight_every; weight_change_inertia=0.9, pde_loss_weights=1, bc_loss_weights=1, additional_loss_weights=1) 
    GradientScaleAdaptiveLoss{Float64}(reweight_every; weight_change_inertia=weight_change_inertia, 
        pde_loss_weights=pde_loss_weights, bc_loss_weights=bc_loss_weights, additional_loss_weights=additional_loss_weights)
end


"""
A way of adaptively reweighting the components of the loss function in the total sum such that the loss weights are maximized by an internal optimiser, which leads to a behavior where loss functions that have not been satisfied get a greater weight,

* `reweight_every`: how often to reweight the PDE and BC loss functions, measured in iterations.  reweighting is cheap since it re-uses the value of loss functions generated during the main optimisation loop,
* `pde_max_optimiser`: a Flux.Optimise.AbstractOptimiser that is used internally to maximize the weights of the PDE loss functions,
* `bc_max_optimiser`: a Flux.Optimise.AbstractOptimiser that is used internally to maximize the weights of the BC loss functions,
* `pde_loss_weights`: either a scalar (which will be broadcast) or vector the size of the number of PDE equations, which describes the initial weight the respective PDE loss has in the full loss sum,
* `bc_loss_weights`: either a scalar (which will be broadcast) or vector the size of the number of BC equations, which describes the initial weight the respective BC loss has in the full loss sum,
* `additional_loss_weights`: a scalar which describes the weight the additional loss function has in the full loss sum, this is currently not adaptive and will be constant with this adaptive loss,

from paper
Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism
Levi McClenny, Ulisses Braga-Neto
https://arxiv.org/abs/2009.04544
"""
mutable struct MiniMaxAdaptiveLoss{T <: Real, PDE_OPT <: Flux.Optimise.AbstractOptimiser, BC_OPT <: Flux.Optimise.AbstractOptimiser} <: AbstractAdaptiveLoss 
    reweight_every::Int64
    pde_max_optimiser::PDE_OPT
    bc_max_optimiser::BC_OPT
    pde_loss_weights::Vector{T} 
    bc_loss_weights::Vector{T} 
    additional_loss_weights::Vector{T} 
    SciMLBase.@add_kwonly function MiniMaxAdaptiveLoss{T, PDE_OPT, BC_OPT}(reweight_every; pde_max_optimiser=Flux.ADAM(1e-4), bc_max_optimiser=Flux.ADAM(0.5), 
            pde_loss_weights=1, bc_loss_weights=1, additional_loss_weights=1) where {T <: Real, PDE_OPT <: Flux.Optimise.AbstractOptimiser, BC_OPT <: Flux.Optimise.AbstractOptimiser}
        new(convert(Int64, reweight_every), convert(PDE_OPT, pde_max_optimiser), convert(BC_OPT, bc_max_optimiser), 
            vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T), vectorify(additional_loss_weights, T))
    end
end

# default to Float64, ADAM, ADAM
SciMLBase.@add_kwonly function MiniMaxAdaptiveLoss(reweight_every; pde_max_optimiser=Flux.ADAM(1e-4), bc_max_optimiser=Flux.ADAM(0.5), pde_loss_weights=1, bc_loss_weights=1, additional_loss_weights=1)
    MiniMaxAdaptiveLoss{Float64, typeof(pde_max_optimiser), typeof(bc_max_optimiser)}(
        reweight_every; pde_max_optimiser=pde_max_optimiser, bc_max_optimiser=bc_max_optimiser,
        pde_loss_weights=pde_loss_weights, bc_loss_weights=bc_loss_weights, additional_loss_weights=additional_loss_weights)
end
