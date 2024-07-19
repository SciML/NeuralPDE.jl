"""
    generate_training_sets(domains, dx, bcs, _indvars::Array, _depvars::Array)

Returns training sets for equations and boundary condition, that is used for `GridTraining`
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
    get_bounds(domains,bcs,_indvars::Array,_depvars::Array)

Returns pairs with lower and upper bounds for all domains. It is used for all non-grid
training strategy: StochasticTraining, QuasiRandomTraining, QuadratureTraining.
"""
function get_bounds end

function get_bounds(
        domains, eqs, bcs, eltypeθ, v::VariableMap, strategy::QuadratureTraining)
    dict_lower_bound = Dict([d.variables => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([d.variables => supremum(d.domain) for d in domains])
    pde_args = get_argument(eqs, v)

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

    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds = [bcs_lower_bounds, bcs_upper_bounds]
    [pde_bounds, bcs_bounds]
end

function get_bounds(domains, eqs, bcs, eltypeθ, v::VariableMap, strategy)
    dx = 1 / strategy.points
    dict_span = Dict([d.variables => [
                          infimum(d.domain) + dx,
                          supremum(d.domain) - dx
                      ] for d in domains])

    # pde_bounds = [[infimum(d.domain),supremum(d.domain)] for d in domains]
    pde_args = get_argument(eqs, v)
    pde_bounds = map(pde_args) do pde_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        bds[1, :], bds[2, :]
    end

    bound_args = get_argument(bcs, v)
    bcs_bounds = map(bound_args) do bound_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, bound_arg)
        bds = eltypeθ.(bds)
        bds[1, :], bds[2, :]
    end
    return pde_bounds, bcs_bounds
end

# TODO: Get this to work with varmap
function get_numeric_integral(pinnrep::PINNRepresentation)
    @unpack strategy, multioutput, derivative, varmap = pinnrep

    integral = (u, cord, phi, integrating_var_id, integrand_func, lb, ub, θ; strategy = strategy, varmap = varmap) -> begin
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
    prob = symbolic_discretize(pde_system::PDESystem, discretization::AbstractPINN)

`symbolic_discretize` is the lower level interface to `discretize` for inspecting internals.
It transforms a symbolic description of a ModelingToolkit-defined `PDESystem` into a
`PINNRepresentation` which holds the pieces required to build an `OptimizationProblem`
for [Optimization.jl](https://docs.sciml.ai/Optimization/stable) or a Likelihood Function
used for HMC based Posterior Sampling Algorithms [AdvancedHMC.jl](https://turinglang.org/AdvancedHMC.jl/stable/)
which is later optimized upon to give Solution or the Solution Distribution of the PDE.

For more information, see `discretize` and `PINNRepresentation`.
"""
function SciMLBase.symbolic_discretize(pdesys::PDESystem,
        discretization::PhysicsInformedNN)
    cardinalize_eqs!(pdesys)
    eqs = pdesys.eqs
    bcs = pdesys.bcs
    domains = pdesys.domain
    eq_params = pdesys.ps
    defaults = pdesys.defaults
    default_p = eq_params == SciMLBase.NullParameters() ? nothing :
                [defaults[ep] for ep in eq_params]

    chain = discretization.chain
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

    varmap = VariableMap(pdesys, discretization)
    eqdata = EquationData(pdesys, varmap, strategy)

    if isnothing(init_params)
        # Use the initialization of the neural network framework
        # But for Lux, default to Float64
        # This is done because Float64 is almost always better for these applications
        if chain isa AbstractArray
            x = map(chain) do x
                _x = ComponentArrays.ComponentArray(Lux.initialparameters(
                    Random.default_rng(),
                    x))
                Float64.(_x) # No ComponentArray GPU support
            end
            # chain_names = ntuple(i -> depvars(eqs[i].lhs, eqdata), length(chain))
            # @show chain_names
            chain_names = Tuple(Symbol.(pdesys.dvs))
            init_params = ComponentArrays.ComponentArray(NamedTuple{chain_names}(i
            for i in x))
        else
            init_params = Float64.(ComponentArrays.ComponentArray(Lux.initialparameters(
                Random.default_rng(),
                chain)))
        end
    else
        init_params = init_params
    end

    flat_init_params = if init_params isa ComponentArrays.ComponentArray
        init_params
        # elseif multioutput
        #     # @assert length(init_params) == length(depvars)
        #     names = ntuple(i -> depvars(eqs[i].lhs, eqdata), length(init_params))
        #     x = ComponentArrays.ComponentArray(NamedTuple{names}(i for i in init_params))
    else
        ComponentArrays.ComponentArray(init_params)
    end

    flat_init_params = if param_estim == false
        ComponentArrays.ComponentArray(; depvar = flat_init_params)
    else
        ComponentArrays.ComponentArray(; depvar = flat_init_params, p = default_p)
    end

    if phi isa Vector
        for ϕ in phi
            ϕ.st = adapt(parameterless_type(ComponentArrays.getdata(flat_init_params)),
                ϕ.st)
        end
    else
        phi.st = adapt(parameterless_type(ComponentArrays.getdata(flat_init_params)),
            phi.st)
    end

    if multioutput
        # acum = [0; accumulate(+, map(length, init_params))]
        phi = map(enumerate(pdesys.dvs)) do (i, dv)
            (coord, expr_θ) -> phi[i](coord, expr_θ.depvar.$(dv))
        end
    else
        # phimap = nothing
        phi = (coord, expr_θ) -> phi(coord, expr_θ.depvar)
    end

    eltypeθ = eltype(flat_init_params)

    if adaloss === nothing
        adaloss = NonAdaptiveLoss{eltypeθ}()
    end

    eqs = map(eq -> eq.lhs, eqs)
    bcs = map(bc -> bc.lhs, bcs)

    pinnrep = PINNRepresentation(eqs, bcs, domains, eq_params, defaults, default_p,
        param_estim, additional_loss, adaloss, varmap, logger,
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
    num_additional_loss = isnothing(additional_loss) ? 0 : 1

    adaloss_T = eltype(adaloss.pde_loss_weights)

    # this will error if the user has provided a number of initial weights that is more than 1 and doesn't match the number of loss functions
    adaloss.pde_loss_weights = ones(adaloss_T, num_pde_losses) .* adaloss.pde_loss_weights
    adaloss.bc_loss_weights = ones(adaloss_T, num_bc_losses) .* adaloss.bc_loss_weights
    adaloss.additional_loss_weights = ones(adaloss_T, num_additional_loss) .*
                                      adaloss.additional_loss_weights

    reweight_losses_func = generate_adaptive_loss_function(pinnrep, adaloss,
        pde_loss_functions,
        bc_loss_functions)

    function get_likelihood_estimate_function(discretization::PhysicsInformedNN)
        function full_loss_function(θ, p)
            # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
            pde_losses = [pde_loss_function(θ) for pde_loss_function in pde_loss_functions]
            bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

            # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
            # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
            ChainRulesCore.@ignore_derivatives if self_increment
                iteration[1] += 1
            end

            ChainRulesCore.@ignore_derivatives begin
                reweight_losses_func(θ, pde_losses,
                    bc_losses)
            end

            weighted_pde_losses = adaloss.pde_loss_weights .* pde_losses
            weighted_bc_losses = adaloss.bc_loss_weights .* bc_losses

            sum_weighted_pde_losses = sum(weighted_pde_losses)
            sum_weighted_bc_losses = sum(weighted_bc_losses)
            weighted_loss_before_additional = sum_weighted_pde_losses +
                                              sum_weighted_bc_losses

            full_weighted_loss = if additional_loss isa Nothing
                weighted_loss_before_additional
            else
                function _additional_loss(phi, θ)
                    (θ_, p_) = if (param_estim == true)
                        θ.depvar, θ.p
                    else
                        θ, nothing
                    end
                    return additional_loss(phi, θ_, p_)
                end
                weighted_additional_loss_val = adaloss.additional_loss_weights[1] *
                                               _additional_loss(phi, θ)
                weighted_loss_before_additional + weighted_additional_loss_val
            end

            ChainRulesCore.@ignore_derivatives begin
                if iteration[1] % log_frequency == 0
                    logvector(pinnrep.logger, pde_losses, "unweighted_loss/pde_losses",
                        iteration[1])
                    logvector(pinnrep.logger,
                        bc_losses,
                        "unweighted_loss/bc_losses",
                        iteration[1])
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
                end
            end

            return full_weighted_loss
        end

        return full_loss_function
    end

    function get_likelihood_estimate_function(discretization::BayesianPINN)
        dataset_pde, dataset_bc = discretization.dataset

        # required as Physics loss also needed on the discrete dataset domain points
        # data points are discrete and so by default GridTraining loss applies
        # passing placeholder dx with GridTraining, it uses data points irl
        datapde_loss_functions, databc_loss_functions = if (!(dataset_bc isa Nothing) ||
                                                            !(dataset_pde isa Nothing))
            merge_strategy_with_loglikelihood_function(pinnrep,
                GridTraining(0.1),
                datafree_pde_loss_functions,
                datafree_bc_loss_functions, train_sets_pde = dataset_pde, train_sets_bc = dataset_bc)
        else
            (nothing, nothing)
        end

        function full_loss_function(θ, allstd::Vector{Vector{Float64}})
            stdpdes, stdbcs, stdextra = allstd
            # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
            pde_loglikelihoods = [logpdf(Normal(0, stdpdes[i]), pde_loss_function(θ))
                                  for (i, pde_loss_function) in enumerate(pde_loss_functions)]

            bc_loglikelihoods = [logpdf(Normal(0, stdbcs[j]), bc_loss_function(θ))
                                 for (j, bc_loss_function) in enumerate(bc_loss_functions)]

            if !(datapde_loss_functions isa Nothing)
                pde_loglikelihoods += [logpdf(Normal(0, stdpdes[j]), pde_loss_function(θ))
                                       for (j, pde_loss_function) in enumerate(datapde_loss_functions)]
            end

            if !(databc_loss_functions isa Nothing)
                bc_loglikelihoods += [logpdf(Normal(0, stdbcs[j]), bc_loss_function(θ))
                                      for (j, bc_loss_function) in enumerate(databc_loss_functions)]
            end

            # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
            # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
            ChainRulesCore.@ignore_derivatives if self_increment
                iteration[1] += 1
            end

            ChainRulesCore.@ignore_derivatives begin
                reweight_losses_func(θ, pde_loglikelihoods,
                    bc_loglikelihoods)
            end

            weighted_pde_loglikelihood = adaloss.pde_loss_weights .* pde_loglikelihoods
            weighted_bc_loglikelihood = adaloss.bc_loss_weights .* bc_loglikelihoods

            sum_weighted_pde_loglikelihood = sum(weighted_pde_loglikelihood)
            sum_weighted_bc_loglikelihood = sum(weighted_bc_loglikelihood)
            weighted_loglikelihood_before_additional = sum_weighted_pde_loglikelihood +
                                                       sum_weighted_bc_loglikelihood

            full_weighted_loglikelihood = if additional_loss isa Nothing
                weighted_loglikelihood_before_additional
            else
                function _additional_loss(phi, θ)
                    (θ_, p_) = if (param_estim == true)
                        θ.depvar, θ.p
                    else
                        θ, nothing
                    end
                    return additional_loss(phi, θ_, p_)
                end

                _additional_loglikelihood = logpdf(Normal(0, stdextra),
                    _additional_loss(phi, θ))

                weighted_additional_loglikelihood = adaloss.additional_loss_weights[1] *
                                                    _additional_loglikelihood

                weighted_loglikelihood_before_additional + weighted_additional_loglikelihood
            end

            return full_weighted_loglikelihood
        end

        return full_loss_function
    end

    full_loss_function = get_likelihood_estimate_function(discretization)
    pinnrep.loss_functions = PINNLossFunctions(bc_loss_functions, pde_loss_functions,
        full_loss_function, additional_loss,
        datafree_pde_loss_functions,
        datafree_bc_loss_functions)

    return pinnrep
end

"""
    discretize(pdesys::PDESystem, discretization::PhysicsInformedNN)

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
