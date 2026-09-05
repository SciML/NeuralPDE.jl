"""
    build_loss_function(pinnrep::PINNRepresentation, eqs, bc_indvars = nothing; kwargs...)

Returns the native array residual kernel for the supplied PDE or boundary equation.
"""
function build_loss_function(pinnrep::PINNRepresentation, eqs, bc_indvars = nothing; kwargs...)
    return build_runtime_array_residual_function(pinnrep, eqs; local_indvars = bc_indvars)
end

"""
    generate_training_sets(domains, dx, bcs, _indvars::Array, _depvars::Array)

Returns training sets for equations and boundary condition, that is used for GridTraining
strategy.
"""
function generate_training_sets end

function generate_training_sets(
        domains, dx, eqs, bcs, eltypeθ, _indvars::Array,
        _depvars::Array
    )
    _, _, dict_indvars, dict_depvars, _ = get_vars(_indvars, _depvars)
    return generate_training_sets(
        domains, dx, eqs, bcs, eltypeθ, dict_indvars,
        dict_depvars
    )
end

# Generate training set in the domain and on the boundary
function generate_training_sets(
        domains, dx, eqs, bcs, eltypeθ, dict_indvars::Dict,
        dict_depvars::Dict
    )
    dxs = dx isa Array ? dx : fill(dx, length(domains))

    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    dict_var_span = Dict(
        [
            Symbol(d.variables) => infimum(d.domain):dx:supremum(d.domain)
                for (d, dx) in zip(domains, dxs)
        ]
    )

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    bound_vars = get_variables(bcs, dict_indvars, dict_depvars)

    dif = [eltypeθ[] for i in 1:size(domains)[1]]
    for _args in bound_vars, (i, x) in enumerate(_args)

        x isa Number && push!(dif[i], x)
    end
    cord_train_set = collect.(spans)
    bc_data = map(zip(dif, cord_train_set)) do (d, c)
        setdiff(c, d)
    end

    dict_var_span_ = Dict([Symbol(d.variables) => bc for (d, bc) in zip(domains, bc_data)])

    bcs_train_sets = map(bound_args) do bt
        isempty(bt) && return _zero_dimensional_coordinates(eltypeθ)
        span = get.((dict_var_span,), bt, bt)
        return reduce(hcat, vec(map(collect, Iterators.product(span...)))) |>
            EltypeAdaptor{eltypeθ}()
    end

    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    pde_train_sets = map(pde_args) do bt
        isempty(bt) && return _zero_dimensional_coordinates(eltypeθ)
        span = get.((dict_var_span_,), bt, bt)
        return reduce(hcat, vec(map(collect, Iterators.product(span...)))) |>
            EltypeAdaptor{eltypeθ}()
    end

    return [pde_train_sets, bcs_train_sets]
end

"""
    get_bounds(domains, bcs, _indvars::Array, _depvars::Array)

Returns pairs with lower and upper bounds for all domains. It is used for all non-grid
training strategy: StochasticTraining, QuasiRandomTraining, QuadratureTraining.
"""
function get_bounds end

function get_bounds(domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array, strategy)
    _, _, dict_indvars, dict_depvars, _ = get_vars(_indvars, _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(
        domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array,
        strategy::QuadratureTraining
    )
    _, _, dict_indvars, dict_depvars, _ = get_vars(_indvars, _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(
        domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars,
        ::QuadratureTraining
    )
    dict_lower_bound = Dict([Symbol(d.variables) => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => supremum(d.domain) for d in domains])

    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    ϵ = cbrt(eps(eltypeθ))
    eltype_adaptor = EltypeAdaptor{eltypeθ}()

    pde_lower_bounds = map(pde_args) do pd
        span = get.((dict_lower_bound,), pd, pd) |> eltype_adaptor
        return span .+ ϵ
    end
    pde_upper_bounds = map(pde_args) do pd
        span = get.((dict_upper_bound,), pd, pd) |> eltype_adaptor
        return span .+ ϵ
    end
    pde_bounds = [pde_lower_bounds, pde_upper_bounds]

    bound_vars = get_variables(bcs, dict_indvars, dict_depvars)

    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds = [bcs_lower_bounds, bcs_upper_bounds]

    return [pde_bounds, bcs_bounds]
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    dx = 1 / strategy.points
    dict_span = Dict(
        [
            Symbol(d.variables) => [
                infimum(d.domain) + dx, supremum(d.domain) - dx,
            ] for d in domains
        ]
    )

    pde_args = get_argument(eqs, dict_indvars, dict_depvars)
    pde_bounds = map(pde_args) do pde_arg
        isempty(pde_arg) && return (eltypeθ[], eltypeθ[])
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    bcs_bounds = map(bound_args) do bound_arg
        isempty(bound_arg) && return (eltypeθ[], eltypeθ[])
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, bound_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    return pde_bounds, bcs_bounds
end

"""
    get_numeric_integral(pinnrep::PINNRepresentation)

Build the numeric integral callback used by generated NeuralPDE loss functions for
integral terms.
"""
function get_numeric_integral end

function get_numeric_integral(pinnrep::PINNRepresentation)
    (;
        strategy, indvars, depvars, derivative, depvars,
        indvars, dict_indvars, dict_depvars,
    ) = pinnrep

    return (
        u,
        cord,
        phi,
        integrating_var_id,
        integrand_func,
        lb,
        ub,
        θ;
        strategy = strategy,
        indvars = indvars,
        depvars = depvars,
        dict_indvars = dict_indvars,
        dict_depvars = dict_depvars,
    ) -> begin
        function integration_(cord, lb, ub, θ)
            cord_ = cord
            function integrand_(x, p)
                @ignore_derivatives cord_[integrating_var_id] .= x
                return integrand_func(cord_, p, phi, derivative, nothing, u, nothing)
            end
            prob_ = IntegralProblem(integrand_, (lb, ub), θ)
            sol = solve(prob_, CubatureJLh(), reltol = 1.0e-3, abstol = 1.0e-3)[1]

            return sol
        end

        T = eltype(cord)
        lb_ = zeros(T, size(lb)[1], size(cord)[2])
        ub_ = zeros(T, size(ub)[1], size(cord)[2])
        for (i, l) in enumerate(lb)
            if l isa Number
                @ignore_derivatives lb_[i, :] .= l
            else
                @ignore_derivatives lb_[i, :] = l(
                    cord, θ, phi, derivative, nothing, u, nothing
                )
            end
        end
        for (i, u_) in enumerate(ub)
            if u_ isa Number
                @ignore_derivatives ub_[i, :] .= u_
            else
                @ignore_derivatives ub_[i, :] = u_(
                    cord, θ, phi, derivative,
                    nothing, u, nothing
                )
            end
        end
        integration_arr = Matrix{T}(undef, 1, 0)
        for i in 1:size(cord, 2)
            integration_arr = hcat(
                integration_arr,
                integration_(cord[:, i], lb_[:, i], ub_[:, i], θ)
            )
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
used for HMC based Posterior Sampling Algorithms
[AdvancedHMC.jl](https://turinglang.org/AdvancedHMC.jl/stable/) which is later optimized
upon to give Solution or the Solution Distribution of the PDE.

For more information, see `discretize` and `PINNRepresentation`.
"""
function SciMLBase.symbolic_discretize(pde_system::PDESystem, discretization::AbstractPINN)
    (; eqs, bcs, domain) = pde_system
    eq_params = pde_system.ps
    defaults = pde_system.initial_conditions
    (;
        chain, param_estim, additional_loss, multioutput, init_params, phi,
        derivative, strategy, logger, iteration, self_increment,
    ) = discretization
    (; log_frequency) = discretization.log_options
    adaloss = discretization.adaptive_loss

    default_p = eq_params isa SciMLBase.NullParameters ? nothing :
        [Symbolics.value(defaults[ep]) for ep in eq_params]

    depvars, indvars, dict_indvars,
        dict_depvars, dict_depvar_input = get_vars(
        get_ivs(pde_system), get_dvs(pde_system)
    )

    if init_params === nothing
        # Use the initialization of the neural network framework
        # But for Lux, default to Float64
        # This is done because Float64 is almost always better for these applications
        if chain isa AbstractArray
            x = map(chain) do x
                ComponentArray{Float64}(LuxCore.initialparameters(Random.default_rng(), x))
            end
            names = ntuple(i -> depvars[i], length(chain))
            init_params = ComponentArray(NamedTuple{names}(Tuple(x)))
        else
            init_params = ComponentArray{Float64}(
                LuxCore.initialparameters(
                    Random.default_rng(), chain
                )
            )
        end
    end

    flat_init_params = if init_params isa ComponentArray
        init_params
    elseif multioutput
        @assert length(init_params) == length(depvars)
        names = ntuple(i -> depvars[i], length(init_params))
        x = ComponentArray(NamedTuple{names}(Tuple(init_params)))
    else
        ComponentArray(init_params)
    end

    flat_init_params = if !param_estim
        multioutput ? ComponentArray(; depvar = flat_init_params) : flat_init_params
    else
        ComponentArray(; depvar = flat_init_params, p = default_p)
    end

    if length(flat_init_params) == 0 && !Base.isconcretetype(eltype(flat_init_params))
        flat_init_params = ComponentArray(
            convert(AbstractArray{Float64}, getdata(flat_init_params)),
            getaxes(flat_init_params)
        )
    end

    adaloss === nothing && (adaloss = NonAdaptiveLoss{eltype(flat_init_params)}())

    eqs isa Array || (eqs = [eqs])

    pde_indvars = if strategy isa QuadratureTraining
        get_argument(eqs, dict_indvars, dict_depvars)
    else
        get_variables(eqs, dict_indvars, dict_depvars)
    end

    bc_indvars = if strategy isa QuadratureTraining
        get_argument(bcs, dict_indvars, dict_depvars)
    else
        get_variables(bcs, dict_indvars, dict_depvars)
    end

    pde_integration_vars = get_integration_variables(eqs, dict_indvars, dict_depvars)
    bc_integration_vars = get_integration_variables(bcs, dict_indvars, dict_depvars)

    pinnrep = PINNRepresentation(
        eqs, bcs, domain, eq_params, defaults, default_p,
        param_estim, additional_loss, adaloss, depvars, indvars,
        dict_indvars, dict_depvars, dict_depvar_input, logger,
        multioutput, iteration, init_params, flat_init_params, phi,
        derivative,
        strategy, pde_indvars, bc_indvars, pde_integration_vars,
        bc_integration_vars, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing
    )

    pinnrep.integral = get_numeric_integral(pinnrep)

    num_pde_losses = length(eqs)
    num_bc_losses = length(bcs)
    num_additional_loss = convert(Int, additional_loss !== nothing)
    adaloss_T = eltype(adaloss.pde_loss_weights)
    adaloss.pde_loss_weights = ones(adaloss_T, num_pde_losses) .* adaloss.pde_loss_weights
    adaloss.bc_loss_weights = ones(adaloss_T, num_bc_losses) .* adaloss.bc_loss_weights
    adaloss.additional_loss_weights = ones(adaloss_T, num_additional_loss) .*
        adaloss.additional_loss_weights

    if discretization isa PhysicsInformedNN
        try
            symbolic_optimization = build_pinn_optimization_system(pinnrep)
            pinnrep.symbolic_pde_residuals = symbolic_optimization.pde_residuals
            pinnrep.symbolic_bc_residuals = symbolic_optimization.bc_residuals
            pinnrep.symbolic_pde_losses = symbolic_optimization.pde_losses
            pinnrep.symbolic_bc_losses = symbolic_optimization.bc_losses
            pinnrep.symbolic_cost = symbolic_optimization.cost
            pinnrep.optimization_system = symbolic_optimization.system
            pinnrep.optimization_system_data = symbolic_optimization.data
        catch err
            err isa InterruptException && rethrow()
            pinnrep.optimization_system_data = PINNOptimizationSystemData(
                nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                false, sprint(showerror, err)
            )
        end
    end

    symbolic_pde_loss_functions = if discretization isa PhysicsInformedNN
        [
            build_runtime_array_residual_function(
                pinnrep, eq; local_indvars = pde_indvar
            ) for (eq, pde_indvar) in zip(eqs, pde_indvars)
        ]
    else
        [
            build_loss_function(pinnrep, eq, pde_indvar)
                for (eq, pde_indvar) in zip(eqs, pde_indvars)
        ]
    end
    symbolic_bc_loss_functions = if discretization isa PhysicsInformedNN
        [
            build_runtime_array_residual_function(
                pinnrep, bc; local_indvars = bc_indvar
            ) for (bc, bc_indvar) in zip(bcs, bc_indvars)
        ]
    else
        [
            build_loss_function(pinnrep, bc, bc_indvar)
                for (bc, bc_indvar) in zip(bcs, bc_indvars)
        ]
    end
    pinnrep.symbolic_pde_loss_functions = symbolic_pde_loss_functions
    pinnrep.symbolic_bc_loss_functions = symbolic_bc_loss_functions

    datafree_pde_loss_functions = symbolic_pde_loss_functions
    datafree_bc_loss_functions = symbolic_bc_loss_functions
    pde_loss_functions,
        bc_loss_functions = merge_strategy_with_loss_function(
        pinnrep, strategy, datafree_pde_loss_functions, datafree_bc_loss_functions
    )

    reweight_losses_func = generate_adaptive_loss_function(
        pinnrep, adaloss,
        pde_loss_functions, bc_loss_functions
    )

    function get_likelihood_estimate_function(::PhysicsInformedNN)
        function full_loss_function(θ, p)
            # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
            pde_losses = [pde_loss_function(θ) for pde_loss_function in pde_loss_functions]
            bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

            # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
            # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
            @ignore_derivatives if self_increment
                iteration[] += 1
            end

            @ignore_derivatives begin
                reweight_losses_func(θ, pde_losses, bc_losses)
            end

            weighted_pde_losses = adaloss.pde_loss_weights .* pde_losses
            weighted_bc_losses = adaloss.bc_loss_weights .* bc_losses

            sum_weighted_pde_losses = sum(weighted_pde_losses)
            sum_weighted_bc_losses = isempty(weighted_bc_losses) ?
                zero(sum_weighted_pde_losses) : sum(weighted_bc_losses)
            weighted_loss_before_additional = sum_weighted_pde_losses +
                sum_weighted_bc_losses

            full_weighted_loss = if additional_loss isa Nothing
                weighted_loss_before_additional
            else
                (θ_, p_) = param_estim ? (θ.depvar, θ.p) : (θ, nothing)
                _additional_loss = additional_loss(phi, θ_, p_)
                weighted_additional_loss_val = adaloss.additional_loss_weights[1] *
                    _additional_loss
                weighted_loss_before_additional + weighted_additional_loss_val
            end

            @ignore_derivatives begin
                if iteration[] % log_frequency == 0
                    logvector(
                        pinnrep.logger, pde_losses, "unweighted_loss/pde_losses",
                        iteration[]
                    )
                    logvector(
                        pinnrep.logger, bc_losses, "unweighted_loss/bc_losses",
                        iteration[]
                    )
                    logvector(
                        pinnrep.logger, weighted_pde_losses,
                        "weighted_loss/weighted_pde_losses", iteration[]
                    )
                    logvector(
                        pinnrep.logger, weighted_bc_losses,
                        "weighted_loss/weighted_bc_losses", iteration[]
                    )
                    if additional_loss !== nothing
                        logscalar(
                            pinnrep.logger, weighted_additional_loss_val,
                            "weighted_loss/weighted_additional_loss", iteration[]
                        )
                    end
                    logscalar(
                        pinnrep.logger, sum_weighted_pde_losses,
                        "weighted_loss/sum_weighted_pde_losses", iteration[]
                    )
                    logscalar(
                        pinnrep.logger, sum_weighted_bc_losses,
                        "weighted_loss/sum_weighted_bc_losses", iteration[]
                    )
                    logscalar(
                        pinnrep.logger, full_weighted_loss,
                        "weighted_loss/full_weighted_loss", iteration[]
                    )
                    logvector(
                        pinnrep.logger, adaloss.pde_loss_weights,
                        "adaptive_loss/pde_loss_weights", iteration[]
                    )
                    logvector(
                        pinnrep.logger, adaloss.bc_loss_weights,
                        "adaptive_loss/bc_loss_weights", iteration[]
                    )
                end
            end

            return full_weighted_loss
        end

        return full_loss_function
    end

    function get_likelihood_estimate_function(discretization::BayesianPINN)
        dataset_pde, dataset_bc = discretization.dataset

        pde_loss_functions,
            bc_loss_functions = merge_strategy_with_loglikelihood_function(
            pinnrep, strategy,
            datafree_pde_loss_functions, datafree_bc_loss_functions
        )

        # required as Physics loss also needed on the discrete dataset domain points
        # data points are discrete and so by default GridTraining loss applies
        # passing placeholder dx with GridTraining, it uses data points irl
        datapde_loss_functions,
            databc_loss_functions = if dataset_bc !== nothing ||
                dataset_pde !== nothing
            merge_strategy_with_loglikelihood_function(
                pinnrep, GridTraining(0.1),
                datafree_pde_loss_functions, datafree_bc_loss_functions,
                train_sets_pde = dataset_pde, train_sets_bc = dataset_bc
            )
        else
            nothing, nothing
        end

        # this includes losses from dataset domain points as well as discretization points
        function full_loss_function(θ, allstd::Vector{Vector{Float64}})
            stdpdes, stdbcs, stdextra = allstd
            # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
            # SSE FOR LOSS ON GRIDPOINTS not MSE ! i, j depend on number of bcs and eqs
            pde_loglikelihoods = sum(
                [
                    pde_loglike_function(θ, stdpdes[i])
                        for (i, pde_loglike_function) in
                        enumerate(pde_loss_functions)
                ]
            )

            bc_loglikelihoods = sum(
                [
                    bc_loglike_function(θ, stdbcs[j])
                        for (j, bc_loglike_function) in
                        enumerate(bc_loss_functions)
                ]
            )

            # final newloss creation components are similar to this
            if !(datapde_loss_functions isa Nothing)
                pde_loglikelihoods += sum(
                    [
                        pde_loglike_function(θ, stdpdes[j])
                            for (j, pde_loglike_function) in
                            enumerate(datapde_loss_functions)
                    ]
                )
            end

            if !(databc_loss_functions isa Nothing)
                bc_loglikelihoods += sum(
                    [
                        bc_loglike_function(θ, stdbcs[j])
                            for (j, bc_loglike_function) in
                            enumerate(databc_loss_functions)
                    ]
                )
            end

            # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
            # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
            @ignore_derivatives if self_increment
                iteration[] += 1
            end

            @ignore_derivatives begin
                reweight_losses_func(
                    θ, pde_loglikelihoods,
                    bc_loglikelihoods
                )
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
                (θ_, p_) = param_estim ? (θ.depvar, θ.p) : (θ, nothing)
                _additional_loss = additional_loss(phi, θ_, p_)
                _additional_loglikelihood = logpdf(Normal(0, stdextra), _additional_loss)

                weighted_additional_loglikelihood = adaloss.additional_loss_weights[1] *
                    _additional_loglikelihood

                weighted_loglikelihood_before_additional + weighted_additional_loglikelihood
            end

            return full_weighted_loglikelihood
        end

        return full_loss_function
    end

    full_loss_function = get_likelihood_estimate_function(discretization)
    pinnrep.loss_functions = PINNLossFunctions(
        bc_loss_functions, pde_loss_functions,
        full_loss_function, additional_loss, datafree_pde_loss_functions,
        datafree_bc_loss_functions
    )

    return pinnrep
end

"""
    prob = discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)

Transforms a symbolic description of a ModelingToolkit-defined `PDESystem` and generates
an `OptimizationProblem` for [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)
whose solution is the solution to the PDE.
"""
function SciMLBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    pinnrep = symbolic_discretize(pde_system, discretization)
    (; log_frequency) = discretization.log_options

    system_data = pinnrep.optimization_system_data
    if system_data isa PINNOptimizationSystemData && system_data.executable
        generated_problem = Optimization.OptimizationProblem(
            system_data.compiled_system, system_data.operating_point
        )
        generated_cost = generated_problem.f
        coordinate_symbols = Any[]
        append!(coordinate_symbols, system_data.coordinate_parameters.pde)
        append!(coordinate_symbols, system_data.coordinate_parameters.bc)
        coordinate_setters = system_data.coordinate_provider === nothing ? Any[] : [
            SymbolicIndexingInterface.setp(generated_problem, symbol)
                for symbol in coordinate_symbols
        ]
        loss_weight_parameters = system_data.loss_weight_parameters
        pde_weight_setter = loss_weight_parameters.pde === nothing ? nothing :
            SymbolicIndexingInterface.setp(generated_problem, loss_weight_parameters.pde)
        bc_weight_setter = loss_weight_parameters.bc === nothing ? nothing :
            SymbolicIndexingInterface.setp(generated_problem, loss_weight_parameters.bc)
        additional_weight_setter = loss_weight_parameters.additional === nothing ? nothing :
            SymbolicIndexingInterface.setp(
                generated_problem, loss_weight_parameters.additional
            )

        current_generated_parameters = Ref{Any}(nothing)

        function update_coordinate_parameters!(parameters, coordinate_sets)
            for (setter, values) in
                zip(coordinate_setters, Iterators.flatten(coordinate_sets))
                setter(parameters, values)
            end
            return nothing
        end

        function loss_weight_vector(weights, active_index)
            new_weights = zeros(eltype(weights), length(weights))
            active_index === nothing || (new_weights[active_index] = one(eltype(weights)))
            return new_weights
        end

        zero_pde_loss_weights = loss_weight_vector(pinnrep.adaloss.pde_loss_weights, nothing)
        zero_bc_loss_weights = loss_weight_vector(pinnrep.adaloss.bc_loss_weights, nothing)
        zero_additional_loss_weights = loss_weight_vector(
            pinnrep.adaloss.additional_loss_weights, nothing
        )

        function set_loss_weight_parameters!(
                parameters, pde_weights, bc_weights, additional_weights
            )
            pde_weight_setter === nothing ||
                pde_weight_setter(parameters, pde_weights)
            bc_weight_setter === nothing ||
                bc_weight_setter(parameters, bc_weights)
            additional_weight_setter === nothing ||
                additional_weight_setter(parameters, additional_weights)
            return nothing
        end

        adaptive_pde_loss_functions = [
            let i = i,
                pde_weights = loss_weight_vector(pinnrep.adaloss.pde_loss_weights, i),
                bc_weights = zero_bc_loss_weights,
                additional_weights = zero_additional_loss_weights
                function (theta)
                    parameters = current_generated_parameters[]
                    @ignore_derivatives set_loss_weight_parameters!(
                        parameters, pde_weights, bc_weights, additional_weights
                    )
                    return generated_cost(theta, parameters)
                end
            end for i in eachindex(pinnrep.adaloss.pde_loss_weights)
        ]
        adaptive_bc_loss_functions = [
            let i = i,
                pde_weights = zero_pde_loss_weights,
                bc_weights = loss_weight_vector(pinnrep.adaloss.bc_loss_weights, i),
                additional_weights = zero_additional_loss_weights
                function (theta)
                    parameters = current_generated_parameters[]
                    @ignore_derivatives set_loss_weight_parameters!(
                        parameters, pde_weights, bc_weights, additional_weights
                    )
                    return generated_cost(theta, parameters)
                end
            end for i in eachindex(pinnrep.adaloss.bc_loss_weights)
        ]

        reweight_losses_func = generate_adaptive_loss_function(
            pinnrep, pinnrep.adaloss, adaptive_pde_loss_functions,
            adaptive_bc_loss_functions
        )

        function strategy_component_losses(loss_functions, theta)
            return [loss_function(theta) for loss_function in loss_functions]
        end

        function weighted_additional_loss(theta)
            pinnrep.additional_loss === nothing && return nothing

            structured_theta = theta isa ComponentArray ?
                theta : ComponentArray(theta, getaxes(pinnrep.flat_init_params))
            network_theta, equation_parameters = pinnrep.param_estim ?
                (structured_theta.depvar, structured_theta.p) : (structured_theta, nothing)
            return pinnrep.adaloss.additional_loss_weights[1] *
                pinnrep.additional_loss(pinnrep.phi, network_theta, equation_parameters)
        end

        function update_loss_weight_parameters!(parameters)
            set_loss_weight_parameters!(
                parameters, pinnrep.adaloss.pde_loss_weights,
                pinnrep.adaloss.bc_loss_weights,
                pinnrep.adaloss.additional_loss_weights
            )
            return nothing
        end

        function optimization_system_cost(theta, parameters)
            @ignore_derivatives current_generated_parameters[] = parameters
            coordinate_sets = nothing
            if system_data.coordinate_provider !== nothing
                coordinate_sets = @ignore_derivatives system_data.coordinate_provider()
                @ignore_derivatives update_coordinate_parameters!(parameters, coordinate_sets)
            end
            @ignore_derivatives if discretization.self_increment
                discretization.iteration[] += 1
            end
            needs_component_losses = !(pinnrep.adaloss isa NonAdaptiveLoss) ||
                pinnrep.logger !== nothing
            pde_losses = nothing
            bc_losses = nothing
            @ignore_derivatives if needs_component_losses
                pde_losses = strategy_component_losses(adaptive_pde_loss_functions, theta)
                bc_losses = strategy_component_losses(adaptive_bc_loss_functions, theta)
                if !(pinnrep.adaloss isa NonAdaptiveLoss)
                    reweight_losses_func(theta, pde_losses, bc_losses)
                    update_loss_weight_parameters!(parameters)
                end
            end
            @ignore_derivatives update_loss_weight_parameters!(parameters)
            full_weighted_loss = generated_cost(theta, parameters)
            @ignore_derivatives if pinnrep.logger !== nothing &&
                                   discretization.iteration[] % log_frequency == 0
                weighted_pde_losses = pinnrep.adaloss.pde_loss_weights .* pde_losses
                weighted_bc_losses = pinnrep.adaloss.bc_loss_weights .* bc_losses
                sum_weighted_pde_losses = sum(weighted_pde_losses)
                sum_weighted_bc_losses = sum(weighted_bc_losses)
                weighted_additional_loss_val = weighted_additional_loss(theta)

                logvector(
                    pinnrep.logger, pde_losses, "unweighted_loss/pde_losses",
                    discretization.iteration[]
                )
                logvector(
                    pinnrep.logger, bc_losses, "unweighted_loss/bc_losses",
                    discretization.iteration[]
                )
                logvector(
                    pinnrep.logger, weighted_pde_losses,
                    "weighted_loss/weighted_pde_losses", discretization.iteration[]
                )
                logvector(
                    pinnrep.logger, weighted_bc_losses,
                    "weighted_loss/weighted_bc_losses", discretization.iteration[]
                )
                weighted_additional_loss_val === nothing || logscalar(
                    pinnrep.logger, weighted_additional_loss_val,
                    "weighted_loss/weighted_additional_loss", discretization.iteration[]
                )
                logscalar(
                    pinnrep.logger, sum_weighted_pde_losses,
                    "weighted_loss/sum_weighted_pde_losses", discretization.iteration[]
                )
                logscalar(
                    pinnrep.logger, sum_weighted_bc_losses,
                    "weighted_loss/sum_weighted_bc_losses", discretization.iteration[]
                )
                logscalar(
                    pinnrep.logger, full_weighted_loss,
                    "weighted_loss/full_weighted_loss", discretization.iteration[]
                )
                logvector(
                    pinnrep.logger, pinnrep.adaloss.pde_loss_weights,
                    "adaptive_loss/pde_loss_weights", discretization.iteration[]
                )
                logvector(
                    pinnrep.logger, pinnrep.adaloss.bc_loss_weights,
                    "adaptive_loss/bc_loss_weights", discretization.iteration[]
                )
            end
            return full_weighted_loss
        end

        f = OptimizationFunction(
            optimization_system_cost, AutoZygote(); sys = system_data.compiled_system
        )
        return Optimization.OptimizationProblem(
            f, pinnrep.flat_init_params, generated_problem.p
        )
    end

    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function, AutoZygote())
    return Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
