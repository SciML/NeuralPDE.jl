abstract type AbstractAdaptiveLoss end

# Utils
function vectorify(x, t::Type{T}) where {T <: Real}
    convertfunc(y) = convert(t, y)
    returnval = if x isa Vector
        convertfunc.(x)
    else
        t[convertfunc(x)]
    end
end

# Dispatches

"""
```julia
NonAdaptiveLoss{T}(; pde_loss_weights = 1,
                     bc_loss_weights = 1,
                     additional_loss_weights = 1)
```

A way of loss weighting the components of the loss function in the total sum that does not
change during optimization
"""
mutable struct NonAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
    SciMLBase.@add_kwonly function NonAdaptiveLoss{T}(; pde_loss_weights = 1,
                                                      bc_loss_weights = 1,
                                                      additional_loss_weights = 1) where {
                                                                                          T <:
                                                                                          Real
                                                                                          }
        new(vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
            vectorify(additional_loss_weights, T))
    end
end

# default to Float64
SciMLBase.@add_kwonly function NonAdaptiveLoss(; pde_loss_weights = 1, bc_loss_weights = 1,
                                               additional_loss_weights = 1)
    NonAdaptiveLoss{Float64}(; pde_loss_weights = pde_loss_weights,
                             bc_loss_weights = bc_loss_weights,
                             additional_loss_weights = additional_loss_weights)
end

function generate_adaptive_loss_function(pinnrep::PINNRepresentation,
                                         adaloss::NonAdaptiveLoss,
                                         pde_loss_functions, bc_loss_functions)
    function null_nonadaptive_loss(θ, pde_losses, bc_losses)
        nothing
    end
end

"""
```julia
GradientScaleAdaptiveLoss(reweight_every;
                          weight_change_inertia = 0.9,
                          pde_loss_weights = 1,
                          bc_loss_weights = 1,
                          additional_loss_weights = 1)
```

A way of adaptively reweighting the components of the loss function in the total sum such
that BC_i loss weights are scaled by the exponential moving average of
max(|∇pde_loss|)/mean(|∇bc_i_loss|) )

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
mutable struct GradientScaleAdaptiveLoss{T <: Real} <: AbstractAdaptiveLoss
    reweight_every::Int64
    weight_change_inertia::T
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
    SciMLBase.@add_kwonly function GradientScaleAdaptiveLoss{T}(reweight_every;
                                                                weight_change_inertia = 0.9,
                                                                pde_loss_weights = 1,
                                                                bc_loss_weights = 1,
                                                                additional_loss_weights = 1) where {
                                                                                                    T <:
                                                                                                    Real
                                                                                                    }
        new(convert(Int64, reweight_every), convert(T, weight_change_inertia),
            vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
            vectorify(additional_loss_weights, T))
    end
end
# default to Float64
SciMLBase.@add_kwonly function GradientScaleAdaptiveLoss(reweight_every;
                                                         weight_change_inertia = 0.9,
                                                         pde_loss_weights = 1,
                                                         bc_loss_weights = 1,
                                                         additional_loss_weights = 1)
    GradientScaleAdaptiveLoss{Float64}(reweight_every;
                                       weight_change_inertia = weight_change_inertia,
                                       pde_loss_weights = pde_loss_weights,
                                       bc_loss_weights = bc_loss_weights,
                                       additional_loss_weights = additional_loss_weights)
end

function generate_adaptive_loss_function(pinnrep::PINNRepresentation,
                                         adaloss::GradientScaleAdaptiveLoss,
                                         pde_loss_functions, bc_loss_functions)
    weight_change_inertia = adaloss.weight_change_inertia
    iteration = pinnrep.iteration
    adaloss_T = eltype(adaloss.pde_loss_weights)

    function run_loss_gradients_adaptive_loss(θ, pde_losses, bc_losses)
        if iteration[1] % adaloss.reweight_every == 0
            # the paper assumes a single pde loss function, so here we grab the maximum of the maximums of each pde loss function
            pde_grads_maxes = [maximum(abs.(Zygote.gradient(pde_loss_function, θ)[1]))
                               for pde_loss_function in pde_loss_functions]
            pde_grads_max = maximum(pde_grads_maxes)
            bc_grads_mean = [mean(abs.(Zygote.gradient(bc_loss_function, θ)[1]))
                             for bc_loss_function in bc_loss_functions]

            nonzero_divisor_eps = adaloss_T isa Float64 ? Float64(1e-11) :
                                  convert(adaloss_T, 1e-7)
            bc_loss_weights_proposed = pde_grads_max ./
                                       (bc_grads_mean .+ nonzero_divisor_eps)
            adaloss.bc_loss_weights .= weight_change_inertia .*
                                       adaloss.bc_loss_weights .+
                                       (1 .- weight_change_inertia) .*
                                       bc_loss_weights_proposed
            logscalar(pinnrep.logger, pde_grads_max, "adaptive_loss/pde_grad_max",
                      iteration[1])
            logvector(pinnrep.logger, pde_grads_maxes, "adaptive_loss/pde_grad_maxes",
                      iteration[1])
            logvector(pinnrep.logger, bc_grads_mean, "adaptive_loss/bc_grad_mean",
                      iteration[1])
            logvector(pinnrep.logger, adaloss.bc_loss_weights,
                      "adaptive_loss/bc_loss_weights",
                      iteration[1])
        end
        nothing
    end
end

"""
```julia
function MiniMaxAdaptiveLoss(reweight_every;
                             pde_max_optimiser = Flux.ADAM(1e-4),
                             bc_max_optimiser = Flux.ADAM(0.5),
                             pde_loss_weights = 1,
                             bc_loss_weights = 1,
                             additional_loss_weights = 1)
```

A way of adaptively reweighting the components of the loss function in the total sum such
that the loss weights are maximized by an internal optimizer, which leads to a behavior
where loss functions that have not been satisfied get a greater weight,

## Positional Arguments

* `reweight_every`: how often to reweight the PDE and BC loss functions, measured in
  iterations.  Reweighting is cheap since it re-uses the value of loss functions generated
  during the main optimization loop.

## Keyword Arguments

* `pde_max_optimiser`: a Flux.Optimise.AbstractOptimiser that is used internally to
  maximize the weights of the PDE loss functions.
* `bc_max_optimiser`: a Flux.Optimise.AbstractOptimiser that is used internally to maximize
  the weights of the BC loss functions.

## References

Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism
Levi McClenny, Ulisses Braga-Neto
https://arxiv.org/abs/2009.04544
"""
mutable struct MiniMaxAdaptiveLoss{T <: Real,
                                   PDE_OPT <: Flux.Optimise.AbstractOptimiser,
                                   BC_OPT <: Flux.Optimise.AbstractOptimiser} <:
               AbstractAdaptiveLoss
    reweight_every::Int64
    pde_max_optimiser::PDE_OPT
    bc_max_optimiser::BC_OPT
    pde_loss_weights::Vector{T}
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
    SciMLBase.@add_kwonly function MiniMaxAdaptiveLoss{T,
                                                       PDE_OPT, BC_OPT}(reweight_every;
                                                                        pde_max_optimiser = Flux.ADAM(1e-4),
                                                                        bc_max_optimiser = Flux.ADAM(0.5),
                                                                        pde_loss_weights = 1,
                                                                        bc_loss_weights = 1,
                                                                        additional_loss_weights = 1) where {
                                                                                                            T <:
                                                                                                            Real,
                                                                                                            PDE_OPT <:
                                                                                                            Flux.Optimise.AbstractOptimiser,
                                                                                                            BC_OPT <:
                                                                                                            Flux.Optimise.AbstractOptimiser
                                                                                                            }
        new(convert(Int64, reweight_every), convert(PDE_OPT, pde_max_optimiser),
            convert(BC_OPT, bc_max_optimiser),
            vectorify(pde_loss_weights, T), vectorify(bc_loss_weights, T),
            vectorify(additional_loss_weights, T))
    end
end

# default to Float64, ADAM, ADAM
SciMLBase.@add_kwonly function MiniMaxAdaptiveLoss(reweight_every;
                                                   pde_max_optimiser = Flux.ADAM(1e-4),
                                                   bc_max_optimiser = Flux.ADAM(0.5),
                                                   pde_loss_weights = 1,
                                                   bc_loss_weights = 1,
                                                   additional_loss_weights = 1)
    MiniMaxAdaptiveLoss{Float64, typeof(pde_max_optimiser),
                        typeof(bc_max_optimiser)}(reweight_every;
                                                  pde_max_optimiser = pde_max_optimiser,
                                                  bc_max_optimiser = bc_max_optimiser,
                                                  pde_loss_weights = pde_loss_weights,
                                                  bc_loss_weights = bc_loss_weights,
                                                  additional_loss_weights = additional_loss_weights)
end

function generate_adaptive_loss_function(pinnrep::PINNRepresentation,
                                         adaloss::MiniMaxAdaptiveLoss,
                                         pde_loss_functions, bc_loss_functions)
    pde_max_optimiser = adaloss.pde_max_optimiser
    bc_max_optimiser = adaloss.bc_max_optimiser
    iteration = pinnrep.iteration

    function run_minimax_adaptive_loss(θ, pde_losses, bc_losses)
        if iteration[1] % adaloss.reweight_every == 0
            Flux.Optimise.update!(pde_max_optimiser, adaloss.pde_loss_weights,
                                  -pde_losses)
            Flux.Optimise.update!(bc_max_optimiser, adaloss.bc_loss_weights, -bc_losses)
            logvector(pinnrep.logger, adaloss.pde_loss_weights,
                      "adaptive_loss/pde_loss_weights", iteration[1])
            logvector(pinnrep.logger, adaloss.bc_loss_weights,
                      "adaptive_loss/bc_loss_weights",
                      iteration[1])
        end
        nothing
    end
end
