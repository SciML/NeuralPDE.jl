struct AdaptiveActivation{T}
  a::T
  n::T
end

Flux.@functor AdaptiveActivation

(fn::AdaptiveActivation)(x) = (fn.n * fn.a) .* x # to be worked on (for weight tying)


struct NonlinearActivation{T}
  σ::T
end

Flux.@functor NonlinearActivation

(a::NonlinearActivation)(x) = (a.σ).(x)


function AdaptiveActivationFeedForwardNetwork(N::Integer, in::Integer, out::Integer, σ = Identity, n::Integer; nn_param_init = glorot_uniform)
  # another parameter would be the type of adaptive fn to be used
  # N = no. of hidden layers

  a = 1/n # initial a scaled such that n*a=1 ?
  function slope_recovery_loss_func(phi, θ, p)
    # calculate the slope_recovery loss function here as a function of the θ parameters that are generated for this
    # network
    for i in 1:1:length(θ):
      # the loss
      """
      if adaptive_fn_without_slope_recovery
        0
      elseif with_slope_recovery_layerwise
        ...
      elseif neuronwise
        ...
      else
        error
      """

    return regularizer_loss
  end

  layer = Flux.Chain(
    Dense(in, out, σ=identity; bias=true, init=nn_param_init),
    AdaptiveActivation(n, a),
    NonlinearActivation(nonlinearity),
  ) # to be stacked for as many hidden layers specified (N)

  return (network=Flux.Chain(...), loss_func=slope_recovery_loss_func)
end
