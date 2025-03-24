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
function build_symbolic_loss_function(pinnrep::PINNRepresentation, eqs;
        eq_params = SciMLBase.NullParameters(), param_estim = false, default_p = nothing,
        bc_indvars = pinnrep.indvars, integrand = nothing,
        dict_transformation_vars = nothing, transformation_vars = nothing,
        integrating_depvars = pinnrep.depvars)
    (; depvars, dict_depvars, dict_depvar_input, phi, derivative, integral, multioutput, init_params, strategy, eq_params, param_estim, default_p) = pinnrep

    if integrand isa Nothing
        loss_function = parse_equation(pinnrep, eqs)
        this_eq_pair = pair(eqs, depvars, dict_depvars, dict_depvar_input)
        this_eq_indvars = unique(vcat(values(this_eq_pair)...))
    else
        this_eq_pair = Dict(map(
            intvars -> dict_depvars[intvars] => dict_depvar_input[intvars],
            integrating_depvars))
        this_eq_indvars = transformation_vars isa Nothing ?
                          unique(vcat(values(this_eq_pair)...)) : transformation_vars
        loss_function = integrand
    end

    vars = :(cord, $θ, phi, derivative, integral, u, p)
    ex = Expr(:block)
    if multioutput
        θ_nums = Symbol[]
        phi_nums = Symbol[]
        for v in depvars
            num = dict_depvars[v]
            push!(θ_nums, :($(Symbol(:($θ), num))))
            push!(phi_nums, :($(Symbol(:phi, num))))
        end

        expr_θ = Expr[]
        expr_phi = Expr[]

        for i in eachindex(depvars)
            push!(expr_θ, :($θ.depvar.$(depvars[i])))
            push!(expr_phi, :(phi[$i]))
        end

        vars_θ = Expr(:(=), build_expr(:tuple, θ_nums), build_expr(:tuple, expr_θ))
        push!(ex.args, vars_θ)

        vars_phi = Expr(:(=), build_expr(:tuple, phi_nums), build_expr(:tuple, expr_phi))
        push!(ex.args, vars_phi)
    end

    #Add an expression for parameter symbols
    if param_estim == true && eq_params != SciMLBase.NullParameters()
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i, eq_param) in enumerate(eq_params)
            push!(expr_params, :($θ.p[$((i):(i))]))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols),
            build_expr(:tuple, expr_params))
        push!(ex.args, params_eq)
    end

    if eq_params != SciMLBase.NullParameters() && param_estim == false
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i, eq_param) in enumerate(eq_params)
            push!(expr_params, :(ArrayInterface.allowed_getindex(p, ($i):($i))))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols),
            build_expr(:tuple, expr_params))
        push!(ex.args, params_eq)
    end

    eq_pair_expr = Expr[]
    for i in keys(this_eq_pair)
        push!(eq_pair_expr, :($(Symbol(:cord, :($i))) = vcat($(this_eq_pair[i]...))))
    end
    vcat_expr = Expr(:block, :($(eq_pair_expr...)))
    vcat_expr_loss_functions = Expr(:block, vcat_expr, loss_function) # TODO rename

    if strategy isa QuadratureTraining
        indvars_ex = get_indvars_ex(bc_indvars)
        left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs),
            build_expr(:tuple, right_arg_pairs))
    else
        indvars_ex = [:($:cord[[$i], :]) for (i, x) in enumerate(this_eq_indvars)]
        left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs),
            build_expr(:tuple, right_arg_pairs))
    end

    if !(dict_transformation_vars isa Nothing)
        transformation_expr_ = Expr[]
        for (i, u) in dict_transformation_vars
            push!(transformation_expr_, :($i = $u))
        end
        transformation_expr = Expr(:block, :($(transformation_expr_...)))
        vcat_expr_loss_functions = Expr(:block, transformation_expr, vcat_expr,
            loss_function)
    end
    let_ex = Expr(:let, vars_eq, vcat_expr_loss_functions)
    push!(ex.args, let_ex)
    return :(($vars) -> begin
        $ex
    end)
end

"""
    build_loss_function(eqs, indvars, depvars, phi, derivative, init_params;
        bc_indvars=nothing)

Returns the body of loss function, which is the executable Julia function, for the main
equation or boundary condition.
"""
function build_loss_function(pinnrep::PINNRepresentation, eqs, bc_indvars)
    (; eq_params, param_estim, default_p, phi, derivative, integral) = pinnrep

    bc_indvars = bc_indvars === nothing ? pinnrep.indvars : bc_indvars

    expr_loss_function = build_symbolic_loss_function(pinnrep, eqs; bc_indvars, eq_params,
        param_estim, default_p)
    u = get_u()
    _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
    return (cord, θ) -> _loss_function(cord, θ, phi, derivative, integral, u, default_p)
end

"""
    generate_training_sets(domains,dx,bcs,_indvars::Array,_depvars::Array)

Returns training sets for equations and boundary condition, that is used for GridTraining
strategy.
"""
function generate_training_sets end

function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, _indvars::Array,
        _depvars::Array)
    _, _, dict_indvars, dict_depvars, _ = get_vars(_indvars, _depvars)
    return generate_training_sets(domains, dx, eqs, bcs, eltypeθ, dict_indvars,
        dict_depvars)
end

# Generate training set in the domain and on the boundary
function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, dict_indvars::Dict,
        dict_depvars::Dict)
    dxs = dx isa Array ? dx : fill(dx, length(domains))

    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    dict_var_span = Dict([Symbol(d.variables) => infimum(d.domain):dx:supremum(d.domain)
                          for (d, dx) in zip(domains, dxs)])

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

function get_bounds(domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array,
        strategy::QuadratureTraining)
    _, _, dict_indvars, dict_depvars, _ = get_vars(_indvars, _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars,
        ::QuadratureTraining)
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
    dict_span = Dict([Symbol(d.variables) => [
                          infimum(d.domain) + dx, supremum(d.domain) - dx] for d in domains])

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

function get_numeric_integral(pinnrep::PINNRepresentation)
    (; strategy, indvars, depvars, derivative, depvars, indvars, dict_indvars, dict_depvars) = pinnrep

    return (u, cord, phi, integrating_var_id, integrand_func, lb, ub, θ; strategy = strategy, indvars = indvars, depvars = depvars, dict_indvars = dict_indvars, dict_depvars = dict_depvars) -> begin
        function integration_(cord, lb, ub, θ)
            cord_ = cord
            function integrand_(x, p)
                @ignore_derivatives cord_[integrating_var_id] .= x
                return integrand_func(cord_, p, phi, derivative, nothing, u, nothing)
            end
            prob_ = IntegralProblem(integrand_, (lb, ub), θ)
            sol = solve(prob_, CubatureJLh(), reltol = 1e-3, abstol = 1e-3)[1]

            return sol
        end

        lb_ = zeros(size(lb)[1], size(cord)[2])
        ub_ = zeros(size(ub)[1], size(cord)[2])
        for (i, l) in enumerate(lb)
            if l isa Number
                @ignore_derivatives lb_[i, :] .= l
            else
                @ignore_derivatives lb_[i, :] = l(
                    cord, θ, phi, derivative, nothing, u, nothing)
            end
        end
        for (i, u_) in enumerate(ub)
            if u_ isa Number
                @ignore_derivatives ub_[i, :] .= u_
            else
                @ignore_derivatives ub_[i, :] = u_(cord, θ, phi, derivative,
                    nothing, u, nothing)
            end
        end
        integration_arr = Matrix{Float64}(undef, 1, 0)
        for i in 1:size(cord, 2)
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
used for HMC based Posterior Sampling Algorithms
[AdvancedHMC.jl](https://turinglang.org/AdvancedHMC.jl/stable/) which is later optimized
upon to give Solution or the Solution Distribution of the PDE.

For more information, see `discretize` and `PINNRepresentation`.
"""
function SciMLBase.symbolic_discretize(pde_system::PDESystem, discretization::AbstractPINN)
    (; eqs, bcs, domain) = pde_system
    eq_params = pde_system.ps
    defaults = pde_system.defaults
    (; chain, param_estim, additional_loss, multioutput, init_params, phi, derivative, strategy, logger, iteration, self_increment) = discretization
    (; log_frequency) = discretization.log_options
    adaloss = discretization.adaptive_loss

    default_p = eq_params isa SciMLBase.NullParameters ? nothing :
                [defaults[ep] for ep in eq_params]

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(
        pde_system.indvars, pde_system.depvars)

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
            init_params = ComponentArray{Float64}(LuxCore.initialparameters(
                Random.default_rng(), chain))
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
            getaxes(flat_init_params))
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

    pinnrep = PINNRepresentation(eqs, bcs, domain, eq_params, defaults, default_p,
        param_estim, additional_loss, adaloss, depvars, indvars,
        dict_indvars, dict_depvars, dict_depvar_input, logger,
        multioutput, iteration, init_params, flat_init_params, phi,
        derivative,
        strategy, pde_indvars, bc_indvars, pde_integration_vars,
        bc_integration_vars, nothing, nothing, nothing, nothing)

    integral = get_numeric_integral(pinnrep)

    symbolic_pde_loss_functions = [build_symbolic_loss_function(pinnrep, eq;
                                       bc_indvars = pde_indvar)
                                   for (eq, pde_indvar) in zip(eqs, pde_indvars,
        pde_integration_vars)]

    symbolic_bc_loss_functions = [build_symbolic_loss_function(pinnrep, bc;
                                      bc_indvars = bc_indvar)
                                  for (bc, bc_indvar) in zip(bcs, bc_indvars,
        bc_integration_vars)]

    pinnrep.integral = integral
    pinnrep.symbolic_pde_loss_functions = symbolic_pde_loss_functions
    pinnrep.symbolic_bc_loss_functions = symbolic_bc_loss_functions

    datafree_pde_loss_functions = [build_loss_function(pinnrep, eq, pde_indvar)
                                   for (eq, pde_indvar) in zip(eqs, pde_indvars)]

    datafree_bc_loss_functions = [build_loss_function(pinnrep, bc, bc_indvar)
                                  for (bc, bc_indvar) in zip(bcs, bc_indvars)]

    pde_loss_functions, bc_loss_functions = merge_strategy_with_loss_function(pinnrep,
        strategy, datafree_pde_loss_functions, datafree_bc_loss_functions)

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

    reweight_losses_func = generate_adaptive_loss_function(pinnrep, adaloss,
        pde_loss_functions, bc_loss_functions)

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
            sum_weighted_bc_losses = sum(weighted_bc_losses)
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
                    logvector(pinnrep.logger, pde_losses, "unweighted_loss/pde_losses",
                        iteration[])
                    logvector(pinnrep.logger, bc_losses, "unweighted_loss/bc_losses",
                        iteration[])
                    logvector(pinnrep.logger, weighted_pde_losses,
                        "weighted_loss/weighted_pde_losses", iteration[])
                    logvector(pinnrep.logger, weighted_bc_losses,
                        "weighted_loss/weighted_bc_losses", iteration[])
                    if additional_loss !== nothing
                        logscalar(pinnrep.logger, weighted_additional_loss_val,
                            "weighted_loss/weighted_additional_loss", iteration[])
                    end
                    logscalar(pinnrep.logger, sum_weighted_pde_losses,
                        "weighted_loss/sum_weighted_pde_losses", iteration[])
                    logscalar(pinnrep.logger, sum_weighted_bc_losses,
                        "weighted_loss/sum_weighted_bc_losses", iteration[])
                    logscalar(pinnrep.logger, full_weighted_loss,
                        "weighted_loss/full_weighted_loss", iteration[])
                    logvector(pinnrep.logger, adaloss.pde_loss_weights,
                        "adaptive_loss/pde_loss_weights", iteration[])
                    logvector(pinnrep.logger, adaloss.bc_loss_weights,
                        "adaptive_loss/bc_loss_weights", iteration[])
                end
            end

            return full_weighted_loss
        end

        return full_loss_function
    end

    function get_likelihood_estimate_function(discretization::BayesianPINN)
        dataset_pde, dataset_bc = discretization.dataset

        pde_loss_functions, bc_loss_functions = merge_strategy_with_loglikelihood_function(
            pinnrep, strategy,
            datafree_pde_loss_functions, datafree_bc_loss_functions)

        # required as Physics loss also needed on the discrete dataset domain points
        # data points are discrete and so by default GridTraining loss applies
        # passing placeholder dx with GridTraining, it uses data points irl
        datapde_loss_functions, databc_loss_functions = if dataset_bc !== nothing ||
                                                           dataset_pde !== nothing
            merge_strategy_with_loglikelihood_function(pinnrep, GridTraining(0.1),
                datafree_pde_loss_functions, datafree_bc_loss_functions,
                train_sets_pde = dataset_pde, train_sets_bc = dataset_bc)
        else
            nothing, nothing
        end

        # this includes losses from dataset domain points as well as discretization points
        function full_loss_function(θ, allstd::Vector{Vector{Float64}})
            stdpdes, stdbcs, stdextra = allstd
            # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
            # SSE FOR LOSS ON GRIDPOINTS not MSE ! i, j depend on number of bcs and eqs
            pde_loglikelihoods = sum([pde_loglike_function(θ, stdpdes[i])
                                      for (i, pde_loglike_function) in enumerate(pde_loss_functions)])

            bc_loglikelihoods = sum([bc_loglike_function(θ, stdbcs[j])
                                     for (j, bc_loglike_function) in enumerate(bc_loss_functions)])

            # final newloss creation components are similar to this
            if !(datapde_loss_functions isa Nothing)
                pde_loglikelihoods += sum([pde_loglike_function(θ, stdpdes[j])
                                           for (j, pde_loglike_function) in enumerate(datapde_loss_functions)])
            end

            if !(databc_loss_functions isa Nothing)
                bc_loglikelihoods += sum([bc_loglike_function(θ, stdbcs[j])
                                          for (j, bc_loglike_function) in enumerate(databc_loss_functions)])
            end

            # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
            # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
            @ignore_derivatives if self_increment
                iteration[] += 1
            end

            @ignore_derivatives begin
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
    pinnrep.loss_functions = PINNLossFunctions(bc_loss_functions, pde_loss_functions,
        full_loss_function, additional_loss, datafree_pde_loss_functions,
        datafree_bc_loss_functions)

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
