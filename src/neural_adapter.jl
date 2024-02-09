function generate_training_sets(domains, dx, eqs, eltypeθ)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx, length(domains))
    end
    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    train_set = adapt(eltypeθ,
                      hcat(vec(map(points -> collect(points), Iterators.product(spans...)))...))
end

function get_loss_function_(loss, init_params, pde_system, strategy::GridTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    eltypeθ = eltype(init_params)
    dx = strategy.dx
    train_set = generate_training_sets(domains, dx, eqs, eltypeθ)
    get_loss_function(loss, train_set, eltypeθ, strategy)
end

function get_bounds_(domains, eqs, eltypeθ, varmap, strategy)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])
    args = get_argument(eqs, varmap)

    bounds = map(args) do pd
        span = map(p -> get(dict_span, p, p), pd)
        map(s -> adapt(eltypeθ, s), span)
    end
    bounds
end

function get_loss_function_(loss, init_params, pde_system, varmap, strategy::StochasticTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    eltypeθ = eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, varmap, strategy)[1]

    get_loss_function(loss, bound, eltypeθ, strategy)
end

function get_loss_function_(loss, init_params, pde_system, varmap, strategy::QuasiRandomTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    eltypeθ = eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, varmap, strategy)[1]

    get_loss_function(loss, bound, eltypeθ, strategy)
end

function get_bounds_(domains, eqs, eltypeθ, varmap,
                     strategy::QuadratureTraining)
    dict_lower_bound = Dict([d.variables => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([d.variables => supremum(d.domain) for d in domains])

    args = get_argument(eqs, varmap)

    lower_bounds = map(args) do pd
        span = map(p -> get(dict_lower_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s), span)
    end
    upper_bounds = map(args) do pd
        span = map(p -> get(dict_upper_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s), span)
    end
    bound = lower_bounds, upper_bounds
end

function get_loss_function_(loss, init_params, pde_system, varmap, strategy::QuadratureTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    eltypeθ = eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, varmap, strategy)
    lb, ub = bound
    get_loss_function(loss, lb[1], ub[1], eltypeθ, strategy)
end

"""
    neural_adapter(loss, init_params, pde_system, strategy)

Trains a neural network using the results from one already obtained prediction.

## Positional Arguments

* `loss`: the body of loss function,
* `init_params`: the initial parameter of the neural network,
* `pde_system`: PDEs are defined using the ModelingToolkit.jl,
* `strategy`: determines which training strategy will be used.
"""
function neural_adapter end

function neural_adapter(loss, init_params, pde_system, strategy)
    varmap = VariableMap(pde_system)
    loss_function__ = get_loss_function_(loss, init_params, pde_system, varmap, strategy)

    function loss_function_(θ, p)
        loss_function__(θ)
    end
    f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(f_, init_params)
end

function neural_adapter(losses::Array, init_params, pde_systems::Array, strategy)
    varmaps = VariableMap.(pde_systems)
    loss_functions_ = map(zip(losses, pde_systems, varmaps)) do (l, p, v)
        get_loss_function_(l, init_params, p, v, strategy)
    end
    loss_function__ = θ -> sum(map(l -> l(θ), loss_functions_))
    function loss_function_(θ, p)
        loss_function__(θ)
    end

    f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(f_, init_params)
end
