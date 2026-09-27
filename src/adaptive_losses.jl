"""
    AbstractAdaptiveLoss

Supertype for callbacks that adapt the symbolic cost weights of an optimization problem.

Pass the returned callback to `solve`. Prefer `save_best = false` with
`OptimizationOptimisers` algorithms: the default `save_best = true` stores the iterate
with the lowest *weighted* objective, but objectives under changing weights are not
comparable, and on the final iteration `save_best` reverts `θ` and re-invokes the
callback at the same `state.iter` (which these rules skip via `last_updated`).
"""
abstract type AbstractAdaptiveLoss end

function _check_update_period(every)
    every > 0 || throw(ArgumentError("`every` must be a positive integer."))
    return Int(every)
end

"""
Return `true` when `state.iter` is due for a weight update. Skips a second call at the
same iteration (the OptimizationOptimisers `save_best` finalization path).
"""
function _due(rule, state)
    state.iter % rule.every == 0 || return false
    rule.last_updated == state.iter && return false
    rule.last_updated = state.iter
    return true
end

"""
    GradientScaleAdaptiveLoss(prob, weights; every = 1, inertia = 0.9, epsilon = 1e-11)

Return an Optimization `solve` callback that updates symbolic cost `weights` from their
parameter-gradient magnitudes, following Wang, Teng, and Perdikaris (2020),
[Understanding and mitigating gradient pathologies in physics-informed neural networks](https://arxiv.org/abs/2001.04536).

At each update the largest maximum absolute cost gradient is divided by each cost's mean
absolute gradient. `inertia` exponentially averages the proposed weights. `prob` must have
been constructed with `discretize(...; weights = weights)`, and every weight must be a
symbolic parameter of its system. Prefer `save_best = false` when solving (see
[`AbstractAdaptiveLoss`](@ref)).
"""
mutable struct GradientScaleAdaptiveLoss{P, W, T} <: AbstractAdaptiveLoss
    prob::P
    weights::W
    every::Int
    inertia::T
    epsilon::T
    context::Any
    last_updated::Int
end

function GradientScaleAdaptiveLoss(prob, weights; every = 1, inertia = 0.9, epsilon = 1.0e-11)
    0 <= inertia <= 1 || throw(ArgumentError("`inertia` must be between zero and one."))
    return GradientScaleAdaptiveLoss(
        prob, weights, _check_update_period(every), inertia, epsilon, nothing, -1
    )
end

"""
    MiniMaxAdaptiveLoss(prob, weights; every = 1, optimizer = Adam(0.5))

Return a `solve` callback that ascends each loss weight with an `Optimisers` rule, porting
the self-adaptive attention update of McClenny and Braga-Neto (2020),
[Self-Adaptive PINNs](https://arxiv.org/abs/2009.04544). Prefer `save_best = false` when
solving (see [`AbstractAdaptiveLoss`](@ref)).
"""
mutable struct MiniMaxAdaptiveLoss{P, W, O} <: AbstractAdaptiveLoss
    prob::P
    weights::W
    every::Int
    optimizer::O
    context::Any
    state::Any
    last_updated::Int
end

function MiniMaxAdaptiveLoss(prob, weights; every = 1, optimizer = Adam(0.5))
    return MiniMaxAdaptiveLoss(
        prob, weights, _check_update_period(every), optimizer,
        nothing, nothing, -1
    )
end

"""
    SoftAdaptAdaptiveLoss(prob, weights; every = 1, α = 0.1, epsilon = 1e-8)

Return a `solve` callback that weights costs by a softmax of their relative loss changes,
following Heydari, Thompson, and Mehmood (2019),
[SoftAdapt](https://arxiv.org/abs/1912.12355). Prefer `save_best = false` when solving
(see [`AbstractAdaptiveLoss`](@ref)); with SoftAdapt a `save_best` double-fire at the
final iteration can otherwise reset stored relative-rate state.
"""
mutable struct SoftAdaptAdaptiveLoss{P, W, T} <: AbstractAdaptiveLoss
    prob::P
    weights::W
    every::Int
    α::T
    epsilon::T
    context::Any
    previous::Any
    last_updated::Int
end

SoftAdaptAdaptiveLoss(prob, weights; every = 1, α = 0.1, epsilon = 1.0e-8) =
    SoftAdaptAdaptiveLoss(
    prob, weights, _check_update_period(every), α, epsilon, nothing, nothing, -1
)

"""
    ReLoBRaLoAdaptiveLoss(prob, weights; every = 1, α = 0.99, β = 0.9,
                          temperature = 1.0, epsilon = 1e-8, rng = Random.default_rng())

Return a `solve` callback implementing Relative Loss Balancing with Random Lookback,
following Bischof and Kraus (2021),
[Multi-Objective Loss Balancing for Physics-Informed Deep Learning](https://arxiv.org/abs/2110.09813).
`α` controls the published moving average, `β` is the probability of carrying the previous
scalings forward, and `temperature` is the paper's `𝒯`: higher values flatten the softmax
toward uniform weights. Prefer `save_best = false` when solving
(see [`AbstractAdaptiveLoss`](@ref)).
"""
mutable struct ReLoBRaLoAdaptiveLoss{P, W, T, R} <: AbstractAdaptiveLoss
    prob::P
    weights::W
    every::Int
    α::T
    β::T
    temperature::T
    epsilon::T
    rng::R
    context::Any
    initial_losses::Any
    previous_losses::Any
    previous_weights::Any
    last_updated::Int
end

function ReLoBRaLoAdaptiveLoss(
        prob, weights; every = 1, α = 0.99, β = 0.9, temperature = 1.0,
        epsilon = 1.0e-8, rng = Random.default_rng()
    )
    (0 <= α <= 1 && 0 <= β <= 1) || throw(
        ArgumentError("`α` and `β` must be between zero and one.")
    )
    temperature > 0 || throw(ArgumentError("`temperature` must be positive."))
    return ReLoBRaLoAdaptiveLoss(
        prob, weights, _check_update_period(every), α, β, temperature, epsilon, rng,
        nothing, nothing, nothing, nothing, -1
    )
end

function _adaptive_context(prob, weights)
    sys = prob.f.sys
    length(weights) == length(ModelingToolkitBase.get_costs(sys)) || throw(
        ArgumentError("`weights` must have one symbolic parameter per system cost.")
    )
    setweights = map(weights) do w
        setp(sys, w)
    end
    getweights = map(weights) do w
        getp(sys, w)
    end
    costfunctions = map(eachindex(weights)) do i
        onehot = [j == i ? 1.0 : 0.0 for j in eachindex(weights)]
        OptimizationFunction{false}(sys; weights = onehot, adtype = prob.f.adtype)
    end
    return setweights, getweights, costfunctions
end

function _cost_values(functions, u, p)
    return [f.f(u, p) for f in functions]
end

function _softmax_weights(scores)
    shifted = scores .- maximum(scores)
    values = exp.(shifted)
    return length(values) .* values ./ sum(values)
end

function (rule::GradientScaleAdaptiveLoss)(state, loss)
    if _due(rule, state)
        if rule.context === nothing
            rule.context = _adaptive_context(rule.prob, rule.weights)
        end
        setweights, getweights, functions = rule.context
        grads = [only(Zygote.gradient(u -> f.f(u, state.p), state.u)) for f in functions]
        scale = maximum(maximum(abs, g) for g in grads)
        current = [getweights[i](state.p) for i in eachindex(rule.weights)]
        proposed = scale == 0 ? current :
            [scale / (mean(abs, g) + rule.epsilon) for g in grads]
        for i in eachindex(rule.weights)
            setweights[i](state.p, rule.inertia * current[i] + (1 - rule.inertia) * proposed[i])
        end
    end
    return false
end

function (rule::MiniMaxAdaptiveLoss)(state, loss)
    if _due(rule, state)
        if rule.context === nothing
            rule.context = _adaptive_context(rule.prob, rule.weights)
        end
        setweights, getweights, functions = rule.context
        values = _cost_values(functions, state.u, state.p)
        current = [getweights[i](state.p) for i in eachindex(rule.weights)]
        rule.state === nothing && (rule.state = Optimisers.setup(rule.optimizer, current))
        Optimisers.update!(rule.state, current, -values)
        for i in eachindex(rule.weights)
            setweights[i](state.p, current[i])
        end
    end
    return false
end

function (rule::SoftAdaptAdaptiveLoss)(state, loss)
    if _due(rule, state)
        if rule.context === nothing
            rule.context = _adaptive_context(rule.prob, rule.weights)
        end
        setweights, _, functions = rule.context
        values = _cost_values(functions, state.u, state.p)
        if rule.previous !== nothing
            scores = rule.α .* (values .- rule.previous) ./ (rule.previous .+ rule.epsilon)
            updated = _softmax_weights(scores)
            for i in eachindex(rule.weights)
                setweights[i](state.p, updated[i])
            end
        end
        rule.previous = values
    end
    return false
end

function (rule::ReLoBRaLoAdaptiveLoss)(state, loss)
    if _due(rule, state)
        if rule.context === nothing
            rule.context = _adaptive_context(rule.prob, rule.weights)
        end
        setweights, getweights, functions = rule.context
        values = _cost_values(functions, state.u, state.p)
        current_weights = [getweights[i](state.p) for i in eachindex(rule.weights)]
        if rule.initial_losses === nothing
            rule.initial_losses = values
            rule.previous_losses = values
            rule.previous_weights = current_weights
        else
            rho = rand(rule.rng) < rule.β
            initial_balance = _softmax_weights(
                values ./ (rule.temperature .* (rule.initial_losses .+ rule.epsilon))
            )
            previous_balance = _softmax_weights(
                values ./ (rule.temperature .* (rule.previous_losses .+ rule.epsilon))
            )
            updated = rule.α .* (
                rho .* rule.previous_weights .+
                    (1 - rho) .* initial_balance
            ) .+ (1 - rule.α) .* previous_balance
            for i in eachindex(rule.weights)
                setweights[i](state.p, updated[i])
            end
            rule.previous_losses = values
            rule.previous_weights = updated
        end
    end
    return false
end
