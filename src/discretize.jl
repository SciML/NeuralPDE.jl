"""
Build a loss function for a PDE or a boundary condition.

# Examples: System of PDEs:

Take expressions in the form:

[Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
 Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

to

:((cord, θ, phi, derivative, u)->begin
          #= ... =#
          #= ... =#
          begin
              (u1, u2) = (θ.depvar.u1, θ.depvar.u2)
              (phi1, phi2) = (phi[1], phi[2])
              let (x, y) = (cord[1], cord[2])
                  [(+)(derivative(phi1, u, [x, y], [[ε, 0.0]], 1, u1), (*)(4, derivative(phi2, u, [x, y], [[0.0, ε]], 1, u1))) - 0,
                   (+)(derivative(phi2, u, [x, y], [[ε, 0.0]], 1, u2), (*)(9, derivative(phi1, u, [x, y], [[0.0, ε]], 1, u2))) - 0]
              end
          end
      end)

for Lux.AbstractLuxLayer.
"""


"""
    generate_training_sets(domains,dx,bcs,_indvars::Array,_depvars::Array)

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
        span = get.((dict_var_span,), bt, bt)
        return reduce(hcat, vec(map(collect, Iterators.product(span...)))) |>
            EltypeAdaptor{eltypeθ}()
    end

    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    pde_train_sets = map(pde_args) do bt
        span = get.((dict_var_span_,), bt, bt)
        return reduce(hcat, vec(map(collect, Iterators.product(span...)))) |>
            EltypeAdaptor{eltypeθ}()
    end

    return [pde_train_sets, bcs_train_sets]
end

"""
    get_bounds(domains,bcs,_indvars::Array,_depvars::Array)

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
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    bcs_bounds = map(bound_args) do bound_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, bound_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    return pde_bounds, bcs_bounds
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
        ModelingToolkit.get_ivs(pde_system), ModelingToolkit.get_dvs(pde_system)
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
        bc_integration_vars, nothing, nothing, nothing, nothing
    )

    integral = nothing
    pinnrep.integral = integral

    symbolic_pde_loss_functions = nothing
    symbolic_bc_loss_functions = nothing

    # Symbolic PINN parser path (default & single engine)
    parsed_ivs = ModelingToolkit.get_ivs(pde_system)
    parsed_dvs = ModelingToolkit.get_dvs(pde_system)
    parsed_ps = eq_params isa SciMLBase.NullParameters ? Any[] : collect(eq_params)
    parsed_default_p = default_p === nothing ? nothing : collect(default_p)
    n_ivs = length(parsed_ivs)
    n_dvs = length(parsed_dvs)

    depvar_theta_template = hasproperty(flat_init_params, :depvar) ?
        flat_init_params.depvar : flat_init_params
    symbolic_init_params = if multioutput
        [depvar_theta_template[dv] for dv in depvars]
    else
        depvar_theta_template
    end

    neural_specs = _symbolic_pinn_neural_specs(
        chain, n_ivs, n_dvs; init_params = symbolic_init_params
    )
    sym_theta0 = _theta0(neural_specs)

    depvar_theta0 = if depvar_theta_template isa ComponentArray
        ComponentArray(sym_theta0, getaxes(depvar_theta_template))
    else
        sym_theta0
    end

    pde_res_data = [
        symbolic_pinn_residual(
            eq, parsed_ivs, parsed_dvs, neural_specs, parsed_ps;
            epsilon = discretization.epsilon
        ) for eq in eqs
    ]
    pde_residuals = [x[1] for x in pde_res_data]
    pde_integrand_syms = [x[2] for x in pde_res_data]
    pde_integrand_fns = [x[3] for x in pde_res_data]

    bc_res_data = [
        symbolic_pinn_residual(
            bc, parsed_ivs, parsed_dvs, neural_specs, parsed_ps;
            epsilon = discretization.epsilon
        ) for bc in bcs
    ]
    bc_residuals = [x[1] for x in bc_res_data]
    bc_integrand_syms = [x[2] for x in bc_res_data]
    bc_integrand_fns = [x[3] for x in bc_res_data]

    pde_compiled = [
        _compiled_residual(
            pde_residuals[i],
            parsed_ivs,
            neural_specs,
            pde_integrand_syms[i],
            pde_integrand_fns[i];
            eq_params = parsed_ps,
            default_eq_params = parsed_default_p,
        ) for i in 1:length(eqs)
    ]
    bc_compiled = [
        _compiled_residual(
            bc_residuals[i],
            parsed_ivs,
            neural_specs,
            bc_integrand_syms[i],
            bc_integrand_fns[i];
            eq_params = parsed_ps,
            default_eq_params = parsed_default_p,
        ) for i in 1:length(bcs)
    ]

    datafree_pde_loss_functions = [_wrap_as_datafree(f) for f in pde_compiled]
    datafree_bc_loss_functions = [_wrap_as_datafree(f) for f in bc_compiled]

    # Override flat_init_params with symbolic parser initialization while
    # preserving the existing container contract (depvar / p).
    flat_init_params = if !param_estim
        multioutput ? ComponentArray(; depvar = depvar_theta0) : depvar_theta0
    else
        ComponentArray(; depvar = depvar_theta0, p = default_p)
    end
    pinnrep.flat_init_params = flat_init_params

    pinnrep.integral = integral
    pinnrep.symbolic_pde_loss_functions = symbolic_pde_loss_functions
    pinnrep.symbolic_bc_loss_functions = symbolic_bc_loss_functions

    pde_loss_functions,
        bc_loss_functions = merge_strategy_with_loss_function(
        pinnrep,
        strategy, datafree_pde_loss_functions, datafree_bc_loss_functions
    )

    # setup for all adaptive losses
    num_pde_losses = length(pde_loss_functions)
    num_bc_losses = length(bc_loss_functions)
    # assume one single additional loss function if there is one. this means that the user needs to lump all their functions into a single one,
    num_additional_loss = convert(Int, additional_loss !== nothing)

    adaloss_T = eltype(adaloss.pde_loss_weights)

    # this will error if the user has provided a number of initial weights that is more than 1 and doesn't match the number of loss functions
    adaloss.pde_loss_weights = ones(adaloss_T, num_pde_losses) .* adaloss.pde_loss_weights
    adaloss.bc_loss_weights = ones(adaloss_T, num_bc_losses) .* adaloss.bc_loss_weights
    adaloss.additional_loss_weights = ones(adaloss_T, num_additional_loss) .*
        adaloss.additional_loss_weights

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

            sum_weighted_pde_losses = isempty(weighted_pde_losses) ? 0.0 : sum(weighted_pde_losses)
            sum_weighted_bc_losses = isempty(weighted_bc_losses) ? 0.0 : sum(weighted_bc_losses)
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
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function, AutoZygote())
    return Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
