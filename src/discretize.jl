"""
Build a loss function for a PDE or a boundary condition

# Examples: System of PDEs:

Take expressions in the form:

[Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
 Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

to

:((cord, θ, phi, derivative, u)->begin
          #= ... =#
          #= ... =#
          begin
              (θ1, θ2) = (θ[1:33], θ"[34:66])
              (phi1, phi2) = (phi[1], phi[2])
              let (x, y) = (cord[1], cord[2])
                  [(+)(derivative(phi1, u, [x, y], [[ε, 0.0]], 1, θ1), (*)(4, derivative(phi2, u, [x, y], [[0.0, ε]], 1, θ2))) - 0,
                   (+)(derivative(phi2, u, [x, y], [[ε, 0.0]], 1, θ2), (*)(9, derivative(phi1, u, [x, y], [[0.0, ε]], 1, θ1))) - 0]
              end
          end
      end)

for Flux.Chain, and

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

for Lux.AbstractExplicitLayer
"""
function build_symbolic_loss_function(pinnrep::PINNRepresentation, eqs;
                                      eq_params = SciMLBase.NullParameters(),
                                      param_estim = false,
                                      default_p = nothing,
                                      bc_indvars = pinnrep.indvars,
                                      integrand = nothing,
                                      dict_transformation_vars = nothing,
                                      transformation_vars = nothing,
                                      integrating_depvars = pinnrep.depvars)
    @unpack indvars, depvars, dict_indvars, dict_depvars, dict_depvar_input,
    phi, derivative, integral,
    multioutput, init_params, strategy, eq_params,
    param_estim, default_p = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    if integrand isa Nothing
        loss_function = parse_equation(pinnrep, eqs)
        this_eq_pair = pair(eqs, depvars, dict_depvars, dict_depvar_input)
        this_eq_indvars = unique(vcat(values(this_eq_pair)...))
    else
        this_eq_pair = Dict(map(intvars -> dict_depvars[intvars] => dict_depvar_input[intvars],
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

        acum = [0; accumulate(+, map(length, init_params))]
        sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]

        for i in eachindex(depvars)
            if (phi isa Vector && phi[1].f isa Optimisers.Restructure) ||
               (!(phi isa Vector) && phi.f isa Optimisers.Restructure)
                # Flux.Chain
                push!(expr_θ, :($θ[$(sep[i])]))
            else # Lux.AbstractExplicitLayer
                push!(expr_θ, :($θ.depvar.$(depvars[i])))
            end
            push!(expr_phi, :(phi[$i]))
        end

        vars_θ = Expr(:(=), build_expr(:tuple, θ_nums), build_expr(:tuple, expr_θ))
        push!(ex.args, vars_θ)

        vars_phi = Expr(:(=), build_expr(:tuple, phi_nums), build_expr(:tuple, expr_phi))
        push!(ex.args, vars_phi)
    end

    #Add an expression for parameter symbols
    if param_estim == true && eq_params != SciMLBase.NullParameters()
        param_len = length(eq_params)
        last_indx = [0; accumulate(+, map(length, init_params))][end]
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i, eq_param) in enumerate(eq_params)
            if (phi isa Vector && phi[1].f isa Optimisers.Restructure) ||
               (!(phi isa Vector) && phi.f isa Optimisers.Restructure)
                push!(expr_params, :($θ[$((i + last_indx):(i + last_indx))]))
            else
                push!(expr_params, :($θ.p[$((i):(i))]))
            end
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
            push!(expr_params, :(ArrayInterfaceCore.allowed_getindex(p, ($i):($i))))
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
    expr_loss_function = :(($vars) -> begin $ex end)
end

"""
```julia
build_loss_function(eqs, indvars, depvars, phi, derivative, init_params; bc_indvars=nothing)
```

Returns the body of loss function, which is the executable Julia function, for the main
equation or boundary condition.
"""
function build_loss_function(pinnrep::PINNRepresentation, eqs, bc_indvars)
    @unpack eq_params, param_estim, default_p, phi, derivative, integral = pinnrep

    bc_indvars = bc_indvars === nothing ? pinnrep.indvars : bc_indvars

    expr_loss_function = build_symbolic_loss_function(pinnrep, eqs;
                                                      bc_indvars = bc_indvars,
                                                      eq_params = eq_params,
                                                      param_estim = param_estim,
                                                      default_p = default_p)
    u = get_u()
    _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
    loss_function = (cord, θ) -> begin _loss_function(cord, θ, phi, derivative, integral, u,
                                                      default_p) end
    return loss_function
end

"""
```julia
generate_training_sets(domains,dx,bcs,_indvars::Array,_depvars::Array)
```

Returns training sets for equations and boundary condition, that is used for GridTraining
strategy.
"""
function generate_training_sets end

function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, _indvars::Array,
                                _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
    return generate_training_sets(domains, dx, eqs, bcs, eltypeθ, dict_indvars,
                                  dict_depvars)
end

# Generate training set in the domain and on the boundary
function generate_training_sets(domains, dx, eqs, bcs, eltypeθ, dict_indvars::Dict,
                                dict_depvars::Dict)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx, length(domains))
    end

    spans = [infimum(d.domain):dx:supremum(d.domain) for (d, dx) in zip(domains, dxs)]
    dict_var_span = Dict([Symbol(d.variables) => infimum(d.domain):dx:supremum(d.domain)
                          for (d, dx) in zip(domains, dxs)])

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    bound_vars = get_variables(bcs, dict_indvars, dict_depvars)

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

    dict_var_span_ = Dict([Symbol(d.variables) => bc for (d, bc) in zip(domains, bc_data)])

    bcs_train_sets = map(bound_args) do bt
        span = map(b -> get(dict_var_span, b, b), bt)
        _set = adapt(eltypeθ,
                     hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end

    pde_vars = get_variables(eqs, dict_indvars, dict_depvars)
    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

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

function get_bounds(domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array, strategy)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(domains, eqs, bcs, eltypeθ, _indvars::Array, _depvars::Array,
                    strategy::QuadratureTraining)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
    return get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars,
                    strategy::QuadratureTraining)
    dict_lower_bound = Dict([Symbol(d.variables) => infimum(d.domain) for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => supremum(d.domain) for d in domains])

    pde_args = get_argument(eqs, dict_indvars, dict_depvars)

    pde_lower_bounds = map(pde_args) do pd
        span = map(p -> get(dict_lower_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s) + cbrt(eps(eltypeθ)), span)
    end
    pde_upper_bounds = map(pde_args) do pd
        span = map(p -> get(dict_upper_bound, p, p), pd)
        map(s -> adapt(eltypeθ, s) - cbrt(eps(eltypeθ)), span)
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

    [pde_bounds, bcs_bounds]
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    dx = 1 / strategy.points
    dict_span = Dict([Symbol(d.variables) => [
                          infimum(d.domain) + dx,
                          supremum(d.domain) - dx,
                      ] for d in domains])

    # pde_bounds = [[infimum(d.domain),supremum(d.domain)] for d in domains]
    pde_args = get_argument(eqs, dict_indvars, dict_depvars)
    pde_bounds = map(pde_args) do pde_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        bds[1, :], bds[2, :]
    end

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
    bcs_bounds = map(bound_args) do bound_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, bound_arg)
        bds = eltypeθ.(bds)
        bds[1, :], bds[2, :]
    end
    return pde_bounds, bcs_bounds
end

function get_numeric_integral(pinnrep::PINNRepresentation)
    @unpack strategy, indvars, depvars, multioutput, derivative,
    depvars, indvars, dict_indvars, dict_depvars = pinnrep

    integral = (u, cord, phi, integrating_var_id, integrand_func, lb, ub, θ; strategy = strategy, indvars = indvars, depvars = depvars, dict_indvars = dict_indvars, dict_depvars = dict_depvars) -> begin
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
prob = symbolic_discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
```

`symbolic_discretize` is the lower level interface to `discretize` for inspecting internals.
It transforms a symbolic description of a ModelingToolkit-defined `PDESystem` into a
`PINNRepresentation` which holds the pieces required to build an `OptimizationProblem`
for [Optimization.jl](https://docs.sciml.ai/Optimization/stable) whose solution is the solution
to the PDE.

For more information, see `discretize` and `PINNRepresentation`.
"""
function SciMLBase.symbolic_discretize(pde_system::PDESystem,
                                       discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs
    chain = discretization.chain

    domains = pde_system.domain
    eq_params = pde_system.ps
    defaults = pde_system.defaults
    default_p = eq_params == SciMLBase.NullParameters() ? nothing :
                [defaults[ep] for ep in eq_params]

    param_estim = discretization.param_estim
    additional_loss = discretization.additional_loss
    adaloss = discretization.adaptive_loss

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(pde_system.indvars,
                                                                               pde_system.depvars)

    multioutput = discretization.multioutput
    init_params = discretization.init_params

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
                names = ntuple(i -> depvars[i], length(chain))
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

    if (discretization.phi isa Vector && discretization.phi[1].f isa Optimisers.Restructure) ||
       (!(discretization.phi isa Vector) && discretization.phi.f isa Optimisers.Restructure)
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

    eltypeθ = eltype(flat_init_params)

    if adaloss === nothing
        adaloss = NonAdaptiveLoss{eltypeθ}()
    end

    phi = discretization.phi

    if (phi isa Vector && phi[1].f isa Lux.AbstractExplicitLayer)
        for ϕ in phi
            ϕ.st = adapt(parameterless_type(ComponentArrays.getdata(flat_init_params)),
                         ϕ.st)
        end
    elseif (!(phi isa Vector) && phi.f isa Lux.AbstractExplicitLayer)
        phi.st = adapt(parameterless_type(ComponentArrays.getdata(flat_init_params)),
                       phi.st)
    end

    derivative = discretization.derivative
    strategy = discretization.strategy

    logger = discretization.logger
    log_frequency = discretization.log_options.log_frequency
    iteration = discretization.iteration
    self_increment = discretization.self_increment

    if !(eqs isa Array)
        eqs = [eqs]
    end

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

    pinnrep = PINNRepresentation(eqs, bcs, domains, eq_params, defaults, default_p,
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
                                   for (eq, pde_indvar, integration_indvar) in zip(eqs,
                                                                                   pde_indvars,
                                                                                   pde_integration_vars)]

    datafree_bc_loss_functions = [build_loss_function(pinnrep, bc, bc_indvar)
                                  for (bc, bc_indvar, integration_indvar) in zip(bcs,
                                                                                 bc_indvars,
                                                                                 bc_integration_vars)]

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
prob = discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
```

Transforms a symbolic description of a ModelingToolkit-defined `PDESystem` and generates
an `OptimizationProblem` for [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) whose
solution is the solution to the PDE.
"""
function SciMLBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    pinnrep = symbolic_discretize(pde_system, discretization)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
                             Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
