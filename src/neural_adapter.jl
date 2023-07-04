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
    depvars, indvars, dict_indvars, dict_depvars = get_vars(pde_system.indvars,
                                                            pde_system.depvars)
    eltypeθ = eltype(init_params)
    dx = strategy.dx
    train_set = generate_training_sets(domains, dx, eqs, eltypeθ)
    get_loss_function(loss, train_set, eltypeθ, strategy)
end

function get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, strategy)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])
    args = get_argument(eqs, dict_indvars, dict_depvars)

    bounds = map(args) do pd
        span = map(p -> get(dict_span, p, p), pd)
        map(s -> adapt(eltypeθ, s), span)
    end
    bounds
end

function get_loss_function_(loss, init_params, pde_system, strategy::StochasticTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    depvars, indvars, dict_indvars, dict_depvars = get_vars(pde_system.indvars,
                                                            pde_system.depvars)

    eltypeθ = eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, strategy)[1]

    get_loss_function(loss, bound, eltypeθ, strategy)
end

function get_loss_function_(loss, init_params, pde_system, strategy::QuasiRandomTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    depvars, indvars, dict_indvars, dict_depvars = get_vars(pde_system.indvars,
                                                            pde_system.depvars)

    eltypeθ = eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, strategy)[1]

    get_loss_function(loss, bound, eltypeθ, strategy)
end

function get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars,
                     strategy::QuadratureTraining)
    dict_lower_bound = Dict([Symbol(d.variables) => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => supremum(d.domain) for d in domains])

    args = get_argument(eqs, dict_indvars, dict_depvars)

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

function get_loss_function_(loss, init_params, pde_system, strategy::QuadratureTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain

    depvars, indvars, dict_indvars, dict_depvars = get_vars(pde_system.indvars,
                                                            pde_system.depvars)

    eltypeθ = eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, strategy)
    lb, ub = bound
    get_loss_function(loss, lb[1], ub[1], eltypeθ, strategy)
end

"""
```julia
neural_adapter(loss, init_params, pde_system, strategy)
```

Trains a neural network using the results from one already obtained prediction.

## Positional Arguments

* `loss`: the body of loss function,
* `init_params`: the initial parameter of the neural network,
* `pde_system`: PDEs are defined using the ModelingToolkit.jl,
* `strategy`: determines which training strategy will be used.
"""
function neural_adapter end

function neural_adapter(loss, init_params, pde_system, strategy)
    loss_function__ = get_loss_function_(loss, init_params, pde_system, strategy)

    function loss_function_(θ, p)
        loss_function__(θ)
    end
    f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(f_, init_params)
end

function neural_adapter(losses::Array, init_params, pde_systems::Array, strategy)
    loss_functions_ = map(zip(losses, pde_systems)) do (l, p)
        get_loss_function_(l, init_params, p, strategy)
    end
    loss_function__ = θ -> sum(map(l -> l(θ), loss_functions_))
    function loss_function_(θ, p)
        loss_function__(θ)
    end

    f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(f_, init_params)
end
