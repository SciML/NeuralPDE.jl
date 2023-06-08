"""
```julia
generate_training_sets(domains,dx,bcs,_indvars::Array,_depvars::Array)
```

Returns training sets for equations and boundary condition, that is used for GridTraining
strategy.
"""
function generate_training_sets end

# Generate training set in the domain and on the boundary
function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, varmap)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx, length(domains))
    end
    @show dxs
    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    dict_var_span = Dict([d.variables => infimum(d.domain):dx:supremum(d.domain)
                          for (d, dx) in zip(domains, dxs)])

    bound_args = get_argument(bcs, varmap)
    bound_vars = get_variables(bcs, varmap)

    dif = [eltypeθ[] for i in 1:size(domains)[1]]
    for _args in bound_vars
        for (i, x) in enumerate(_args)
            if x isa Number
                push!(dif[i], x)
            end
        end
    end
    cord_train_set = collect.(spans)
    bc_data = map(zip(dif, cord_train_set)) do (d, c)
        setdiff(c, d)
    end

    dict_var_span_ = Dict([d.variables => bc for (d, bc) in zip(domains, bc_data)])

    bcs_train_sets = map(bound_args) do bt
        span = map(b -> get(dict_var_span, b, b), bt)
        _set = adapt(eltypeθ,
                     hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end

    pde_vars = get_variables(eqs, varmap)
    pde_args = get_argument(eqs, varmap)

    pde_train_set = adapt(eltypeθ,
                          hcat(vec(map(points -> collect(points),
                                       Iterators.product(bc_data...)))...))

    pde_train_sets = map(pde_args) do bt
        span = map(b -> get(dict_var_span_, b, b), bt)
        _set = adapt(eltypeθ,
                     hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end
    [pde_train_sets, bcs_train_sets]
end

"""
```julia
get_bounds(domains,bcs,_indvars::Array,_depvars::Array)
```

Returns pairs with lower and upper bounds for all domains. It is used for all non-grid
training strategy: StochasticTraining, QuasiRandomTraining, QuadratureTraining.
"""
function get_bounds end

function get_bounds(domains, eqs, bcs, eltypeθ, v::VariableMap, strategy::AbstractGridfreeStrategy)
    dict_lower_bound = Dict([d.variables => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([d.variables => supremum(d.domain) for d in domains])
    pde_args = get_argument(eqs, v)
    @show pde_args

    pde_lower_bounds = map(pde_args) do pd
        span = map(p -> get(dict_lower_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s) + cbrt(eps(eltypeθ)), span)
    end
    pde_upper_bounds = map(pde_args) do pd
        span = map(p -> get(dict_upper_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s) - cbrt(eps(eltypeθ)), span)
    end
    pde_bounds = [pde_lower_bounds, pde_upper_bounds]

    bound_vars = get_variables(bcs, v)
    @show bound_vars

    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds = [bcs_lower_bounds, bcs_upper_bounds]
    @show bcs_bounds pde_bounds
    [pde_bounds, bcs_bounds]
end
# TODO: Get this to work with varmap
function get_numeric_integral(pinnrep::PINNRepresentation)
    @unpack strategy, multioutput, derivative, varmap = pinnrep

    integral = (u, cord, phi, integrating_var_id, integrand_func, lb, ub, θ; strategy = strategy, varmap=varmap) -> begin
        function integration_(cord, lb, ub, θ)
            cord_ = cord
            function integrand_(x, p)
                ChainRulesCore.@ignore_derivatives @views(cord_[integrating_var_id]) .= x
                return integrand_func(cord_, p, phi, derivative, nothing, u, nothing)
            end
            prob_ = IntegralProblem(integrand_, lb, ub, θ)
            sol = solve(prob_, CubatureJLh(), reltol = 1e-3, abstol = 1e-3)[1]

            return sol
        end

        lb_ = zeros(size(lb)[1], size(cord)[2])
        ub_ = zeros(size(ub)[1], size(cord)[2])
        for (i, l) in enumerate(lb)
            if l isa Number
                ChainRulesCore.@ignore_derivatives lb_[i, :] = fill(l, 1, size(cord)[2])
            else
                ChainRulesCore.@ignore_derivatives lb_[i, :] = l(cord, θ, phi, derivative,
                                                                 nothing, u, nothing)
            end
        end
        for (i, u_) in enumerate(ub)
            if u_ isa Number
                ChainRulesCore.@ignore_derivatives ub_[i, :] = fill(u_, 1, size(cord)[2])
            else
                ChainRulesCore.@ignore_derivatives ub_[i, :] = u_(cord, θ, phi, derivative,
                                                                  nothing, u, nothing)
            end
        end
        integration_arr = Matrix{Float64}(undef, 1, 0)
        for i in 1:size(cord)[2]
            # ub__ = @Zygote.ignore getindex(ub_, :,  i)
            # lb__ = @Zygote.ignore getindex(lb_, :,  i)
            integration_arr = hcat(integration_arr,
                                   integration_(cord[:, i], lb_[:, i], ub_[:, i], θ))
        end
        return integration_arr
    end
end

"""
```julia
prob = symbolic_discretize(pdesys::PDESystem, discretization::PhysicsInformedNN)
```

`symbolic_discretize` is the lower level interface to `discretize` for inspecting internals.
It transforms a symbolic description of a ModelingToolkit-defined `PDESystem` into a
`PINNRepresentation` which holds the pieces required to build an `OptimizationProblem`
for [Optimization.jl](https://docs.sciml.ai/Optimization/stable) whose solution is the solution
to the PDE.

For more information, see `discretize` and `PINNRepresentation`.
"""
function SciMLBase.symbolic_discretize(pdesys::PDESystem,
                                       discretization::PhysicsInformedNN)
    cardinalize_eqs!(pdesys)
    eqs = pdesys.eqs
    bcs = pdesys.bcs
    chain = discretization.chain

    domains = pdesys.domain
    eq_params = pdesys.ps
    defaults = pdesys.defaults
    default_p = eq_params == SciMLBase.NullParameters() ? nothing :
                [defaults[ep] for ep in eq_params]

    param_estim = discretization.param_estim
    additional_loss = discretization.additional_loss
    adaloss = discretization.adaptive_loss


    multioutput = discretization.multioutput
    init_params = discretization.init_params
    phi = discretization.phi

    derivative = discretization.derivative
    strategy = discretization.strategy

    logger = discretization.logger
    log_frequency = discretization.log_options.log_frequency
    iteration = discretization.iteration
    self_increment = discretization.self_increment

    v = VariableMap(pdesys, discretization)

    eqdata = EquationData(pdesys, v, strategy)


    if init_params === nothing
        # Use the initialization of the neural network framework
        # But for Lux, default to Float64
        # For Flux, default to the types matching the values in the neural network
        # This is done because Float64 is almost always better for these applications
        # But with Flux there's already a chosen type from the user

        if chain isa AbstractArray
            if chain[1] isa Flux.Chain
                init_params = map(chain) do x
                    _x = Flux.destructure(x)[1]
                end
            else
                x = map(chain) do x
                    _x = ComponentArrays.ComponentArray(Lux.initialparameters(Random.default_rng(),
                        x))
                    Float64.(_x) # No ComponentArray GPU support
                end
                names = ntuple(i -> Symbol.(v.ū)[i], length(chain))
                init_params = ComponentArrays.ComponentArray(NamedTuple{names}(i
                                                                                for i in x))
            end
        else
            if chain isa Flux.Chain
                init_params = Flux.destructure(chain)[1]
                init_params = init_params isa Array ? Float64.(init_params) :
                                init_params
            else
                init_params = Float64.(ComponentArrays.ComponentArray(Lux.initialparameters(Random.default_rng(),
                    chain)))
            end
        end
    else
        init_params = init_params
    end

    if (phi isa Vector && phi[1].f isa Optimisers.Restructure) ||
        (!(phi isa Vector) && phi.f isa Optimisers.Restructure)
        # Flux.Chain
        flat_init_params = multioutput ? reduce(vcat, init_params) : init_params
        flat_init_params = param_estim == false ? flat_init_params :
                            vcat(flat_init_params,
            adapt(typeof(flat_init_params), default_p))
    else
        flat_init_params = if init_params isa ComponentArrays.ComponentArray
            init_params
        elseif multioutput
            @assert length(init_params) == length(depvars)
            names = ntuple(i -> depvars[i], length(init_params))
            x = ComponentArrays.ComponentArray(NamedTuple{names}(i for i in init_params))
        else
            ComponentArrays.ComponentArray(init_params)
        end
        flat_init_params = if param_estim == false && multioutput
            ComponentArrays.ComponentArray(; depvar = flat_init_params)
        elseif param_estim == false && !multioutput
            flat_init_params
        else
            ComponentArrays.ComponentArray(; depvar = flat_init_params, p = default_p)
        end
    end

    if (phi isa Vector && phi[1].f isa Lux.AbstractExplicitLayer)
        for ϕ in phi
            ϕ.st = adapt(parameterless_type(ComponentArrays.getdata(flat_init_params)),
                ϕ.st)
        end
    elseif (!(phi isa Vector) && phi.f isa Lux.AbstractExplicitLayer)
        phi.st = adapt(parameterless_type(ComponentArrays.getdata(flat_init_params)),
            phi.st)
    end

    eltypeθ = eltype(flat_init_params)

    if adaloss === nothing
        adaloss = NonAdaptiveLoss{eltypeθ}()
    end

    eqs = map(eq -> eq.lhs, eqs)
    bcs = map(bc -> bc.lhs, bcs)

    pinnrep = PINNRepresentation(eqs, bcs, domains, eq_params, defaults, default_p,
                                 param_estim, additional_loss, adaloss, v, logger,
                                 multioutput, iteration, init_params, flat_init_params, phi,
                                 derivative,
                                 strategy, eqdata, nothing, nothing, nothing, nothing)

    #integral = get_numeric_integral(pinnrep)

    #symbolic_pde_loss_functions = [build_symbolic_loss_function(pinnrep, eq) for eq in eqs]

    #symbolic_bc_loss_functions = [build_symbolic_loss_function(pinnrep, bc) |> toexpr for bc in bcs]

    #pinnrep.integral = integral
    #pinnrep.symbolic_pde_loss_functions = symbolic_pde_loss_functions
    #pinnrep.symbolic_bc_loss_functions = symbolic_bc_loss_functions

    datafree_pde_loss_functions = [build_loss_function(pinnrep, eq) for eq in eqs]

    datafree_bc_loss_functions = [build_loss_function(pinnrep, bc) for bc in bcs]

    pde_loss_functions, bc_loss_functions = merge_strategy_with_loss_function(pinnrep,
                                                                              strategy,
                                                                              datafree_pde_loss_functions,
                                                                              datafree_bc_loss_functions)

    # setup for all adaptive losses
    num_pde_losses = length(pde_loss_functions)
    num_bc_losses = length(bc_loss_functions)
    # assume one single additional loss function if there is one. this means that the user needs to lump all their functions into a single one,
    num_additional_loss = additional_loss isa Nothing ? 0 : 1

    adaloss_T = eltype(adaloss.pde_loss_weights)

    # this will error if the user has provided a number of initial weights that is more than 1 and doesn't match the number of loss functions
    adaloss.pde_loss_weights = ones(adaloss_T, num_pde_losses) .* adaloss.pde_loss_weights
    adaloss.bc_loss_weights = ones(adaloss_T, num_bc_losses) .* adaloss.bc_loss_weights
    adaloss.additional_loss_weights = ones(adaloss_T, num_additional_loss) .*
                                      adaloss.additional_loss_weights

    reweight_losses_func = generate_adaptive_loss_function(pinnrep, adaloss,
                                                           pde_loss_functions,
                                                           bc_loss_functions)

    function full_loss_function(θ, p)

        # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
        pde_losses = [pde_loss_function(θ) for pde_loss_function in pde_loss_functions]
        bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

        # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
        # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
        ChainRulesCore.@ignore_derivatives if self_increment
            iteration[1] += 1
        end

        ChainRulesCore.@ignore_derivatives begin reweight_losses_func(θ, pde_losses,
                                                                      bc_losses) end

        weighted_pde_losses = adaloss.pde_loss_weights .* pde_losses
        weighted_bc_losses = adaloss.bc_loss_weights .* bc_losses

        sum_weighted_pde_losses = sum(weighted_pde_losses)
        sum_weighted_bc_losses = sum(weighted_bc_losses)
        weighted_loss_before_additional = sum_weighted_pde_losses + sum_weighted_bc_losses

        full_weighted_loss = if additional_loss isa Nothing
            weighted_loss_before_additional
        else
            function _additional_loss(phi, θ)
                (θ_, p_) = if (param_estim == true)
                    if (phi isa Vector && phi[1].f isa Optimisers.Restructure) ||
                       (!(phi isa Vector) && phi.f isa Optimisers.Restructure)
                        # Isa Flux Chain
                        θ[1:(end - length(default_p))], θ[(end - length(default_p) + 1):end]
                    else
                        θ.depvar, θ.p
                    end
                else
                    θ, nothing
                end
                return additional_loss(phi, θ_, p_)
            end
            weighted_additional_loss_val = adaloss.additional_loss_weights[1] *
                                           _additional_loss(phi, θ)
            weighted_loss_before_additional + weighted_additional_loss_val
        end

        ChainRulesCore.@ignore_derivatives begin if iteration[1] % log_frequency == 0
            logvector(pinnrep.logger, pde_losses, "unweighted_loss/pde_losses",
                      iteration[1])
            logvector(pinnrep.logger, bc_losses, "unweighted_loss/bc_losses", iteration[1])
            logvector(pinnrep.logger, weighted_pde_losses,
                      "weighted_loss/weighted_pde_losses",
                      iteration[1])
            logvector(pinnrep.logger, weighted_bc_losses,
                      "weighted_loss/weighted_bc_losses",
                      iteration[1])
            if !(additional_loss isa Nothing)
                logscalar(pinnrep.logger, weighted_additional_loss_val,
                          "weighted_loss/weighted_additional_loss", iteration[1])
            end
            logscalar(pinnrep.logger, sum_weighted_pde_losses,
                      "weighted_loss/sum_weighted_pde_losses", iteration[1])
            logscalar(pinnrep.logger, sum_weighted_bc_losses,
                      "weighted_loss/sum_weighted_bc_losses", iteration[1])
            logscalar(pinnrep.logger, full_weighted_loss,
                      "weighted_loss/full_weighted_loss",
                      iteration[1])
            logvector(pinnrep.logger, adaloss.pde_loss_weights,
                      "adaptive_loss/pde_loss_weights",
                      iteration[1])
            logvector(pinnrep.logger, adaloss.bc_loss_weights,
                      "adaptive_loss/bc_loss_weights",
                      iteration[1])
        end end

        return full_weighted_loss
    end

    pinnrep.loss_functions = PINNLossFunctions(bc_loss_functions, pde_loss_functions,
                                               full_loss_function, additional_loss,
                                               datafree_pde_loss_functions,
                                               datafree_bc_loss_functions)

    return pinnrep
end

"""
```julia
prob = discretize(pdesys::PDESystem, discretization::PhysicsInformedNN)
```

Transforms a symbolic description of a ModelingToolkit-defined `PDESystem` and generates
an `OptimizationProblem` for [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) whose
solution is the solution to the PDE.
"""
function SciMLBase.discretize(pdesys::PDESystem, discretization::PhysicsInformedNN)
    pinnrep = symbolic_discretize(pdesys, discretization)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
                             Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
