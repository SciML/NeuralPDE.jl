function generate_training_sets(domains, dx, eqs, eltypeθ)
    dxs = dx isa Array ? dx : fill(dx, length(domains))
    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    return reduce(hcat, vec(map(collect, Iterators.product(spans...)))) |>
        EltypeAdaptor{eltypeθ}()
end

function get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, _)
    dict_span = Dict(
        [
            Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                for d in domains
        ]
    )
    args = get_argument(eqs, dict_indvars, dict_depvars)

    bounds = first(
        map(args) do pd
            return get.((dict_span,), pd, pd) |> EltypeAdaptor{eltypeθ}()
        end
    )
    return first.(bounds), last.(bounds)
end

function get_loss_function_neural_adapter(
        loss, init_params, pde_system, strategy::GridTraining
    )
    eqs = pde_system.eqs
    eqs isa Array || (eqs = [eqs])
    eltypeθ = recursive_eltype(init_params)
    train_set = generate_training_sets(pde_system.domain, strategy.dx, eqs, eltypeθ)
    return get_loss_function(init_params, loss, train_set, eltypeθ, strategy)
end

function get_loss_function_neural_adapter(
        loss, init_params, pde_system,
        strategy::Union{StochasticTraining, QuasiRandomTraining}
    )
    eqs = pde_system.eqs
    eqs isa Array || (eqs = [eqs])
    domains = pde_system.domain

    _, _, dict_indvars, dict_depvars = get_vars(pde_system.indvars, pde_system.depvars)

    eltypeθ = recursive_eltype(init_params)
    bound = get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, strategy)
    return get_loss_function(init_params, loss, bound, eltypeθ, strategy)
end

function get_loss_function_neural_adapter(
        loss, init_params, pde_system, strategy::QuadratureTraining
    )
    eqs = pde_system.eqs
    eqs isa Array || (eqs = [eqs])
    domains = pde_system.domain

    _, _, dict_indvars, dict_depvars = get_vars(pde_system.indvars, pde_system.depvars)

    eltypeθ = recursive_eltype(init_params)
    lb, ub = get_bounds_(domains, eqs, eltypeθ, dict_indvars, dict_depvars, strategy)
    return get_loss_function(init_params, loss, lb, ub, eltypeθ, strategy)
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
    loss_function = get_loss_function_neural_adapter(
        loss, init_params, pde_system, strategy
    )
    return OptimizationProblem(
        OptimizationFunction((θ, _) -> loss_function(θ), AutoZygote()), init_params
    )
end

function neural_adapter(losses::Array, init_params, pde_systems::Array, strategy)
    loss_functions = map(zip(losses, pde_systems)) do (l, p)
        get_loss_function_neural_adapter(l, init_params, p, strategy)
    end
    return OptimizationProblem(
        OptimizationFunction((θ, _) -> sum(l -> l(θ), loss_functions), AutoZygote()),
        init_params
    )
end
