# TODO: add multioutput
# TODO: add integrals

function build_symbolic_loss_function(pinnrep::PINNRepresentation, eq;
        eq_params = SciMLBase.NullParameters(),
        param_estim = false,
        default_p = [],
        integrand = nothing,
        transformation_vars = nothing)
    @unpack varmap, eqdata,
    phi, derivative, integral,
    multioutput, init_params, strategy,
    param_estim, default_p = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    eq = eq isa Equation ? eq.lhs : eq

    eq_args = get(eqdata.ivargs, eq, varmap.x̄)

    if isnothing(integrand)
        this_eq_indvars = indvars(eq, eqdata)
        this_eq_depvars = depvars(eq, eqdata)
        loss_function = parse_equation(pinnrep, eq, eq_iv_args(eq, eqdata))
    else
        this_eq_indvars = transformation_vars isa Nothing ?
                          unique(indvars(eq, eqmap)) : transformation_vars
        loss_function = integrand
    end

    n = length(this_eq_indvars)

    get_ps = if param_estim == true && !isnothing(default_p)
        (θ) -> θ.p[1:length(eq_params)]
    else
        (θ) -> default_p
    end

    function get_coords(cord)
        num_numbers = 0
        out = map(enumerate(eq_args)) do (i, x)
            if x isa Number
                fill(convert(eltypeθ, x), size(cord[[1], :]))
            else
                cord[[i], :]
            end
        end
        if out === nothing
            return []
        else
            return out
        end
    end

    full_loss_func = (cord, θ, phi, p) -> begin
        coords = [[nothing]]
        @ignore_derivatives coords = get_coords(cord)
        @show coords
        loss_function(coords, θ, phi, get_ps(θ))
    end
    return full_loss_func
end

@register_array_symbolic (f::Phi{<:Lux.AbstractExplicitLayer})(
    x::AbstractArray, ps::Union{NamedTuple, <:AbstractVector}) begin
    size = LuxCore.outputsize(f.f, x, LuxCore._default_rng())
    eltype = Real
end

function build_loss_function(pinnrep, eq)
    @unpack eq_params, param_estim, default_p, phi, multioutput, derivative, integral = pinnrep
    _loss_function = build_symbolic_loss_function(pinnrep, eq,
        eq_params = eq_params,
        param_estim = param_estim)
    loss_function = (cord, θ) -> begin
        _loss_function(cord, θ, phi,
            default_p)
    end
    return loss_function
end

function operations(ex)
    if istree(ex)
        op = operation(ex)
        return vcat(operations.(arguments(ex))..., op)
    end
    return []
end

############################################################################################
# Parse equation
############################################################################################

function parse_equation(pinnrep::PINNRepresentation, term, ivs; is_integral = false,
        dict_transformation_vars = nothing,
        transformation_vars = nothing)
    @unpack varmap, eqdata, derivative, integral, flat_init_params, phi, depvars_outs_map, = pinnrep
    eltypeθ = eltype(flat_init_params)

    ex_vars = get_depvars(term, varmap.depvar_ops)

    # if multioutput
    #     dummyvars = @variables switch
    # else
    #     dummyvars = @variables switch
    # end
    dummyvars = @variables switch

    dummyvars = unwrap.(dummyvars)
    deriv_rules = generate_derivative_rules(
        term, eqdata, eltypeθ, dummyvars, derivative, varmap, depvars_outs_map)
    ch = Prewalk(Chain(deriv_rules))

    expr = ch(term)
    #expr = swch(expr)

    sym_coords = DestructuredArgs(ivs)
    ps = DestructuredArgs(varmap.ps)

    args = [sym_coords, ps]

    ex = Func(args, [], expr) |> toexpr |> _dot_

    @show ex
    f = @RuntimeGeneratedFunction ex
    return f
end

function generate_derivative_rules(
        term, eqdata, eltypeθ, dummyvars, derivative, varmap, depvars_outs_map)
    switch = dummyvars
    # if symtype(phi) isa AbstractArray
    #     phi = collect(phi)
    # end

    dvs = get_depvars(term, varmap.depvar_ops)

    # Orthodox derivatives
    n(w) = length(arguments(w))
    rs = reduce(vcat,
        [reduce(vcat,
             [[@rule $((Differential(x)^d)(w)) => derivative(
                   depvars_outs_map[operation(w)], arguments(w),
                   get_ε(n(w), j, eltypeθ, d),
                   d, θ)
               for d in differential_order(term, x)]
              for (j, x) in enumerate(varmap.args[operation(w)])],
             init = [])
         for w in dvs],
        init = [])

    # Mixed derivatives
    mx = mapreduce(vcat, dvs, init = []) do w
        mapreduce(vcat, enumerate(varmap.args[operation(w)]), init = []) do (j, x)
            mapreduce(vcat, enumerate(varmap.args[operation(w)]), init = []) do (k, y)
                if isequal(x, y)
                    [(_) -> nothing]
                else
                    ε1 = get_ε(n(w), j, eltypeθ, 1)
                    ε2 = get_ε(n(w), k, eltypeθ, 1)
                    [@rule $((Differential(x))((Differential(y))(w))) => derivative(
                        (coord_, θ_) -> derivative(
                            depvars_outs_map[operation(w)], arguments(w),
                            ε2, 1, θ_),
                        arguments(w), ε1, 1, θ)]
                end
            end
        end
    end

    vr = mapreduce(vcat, dvs, init = []) do w
        @rule w => depvars_outs_map[operation(w)](arguments(w))
    end

    return [mx; rs; vr]
end

function generate_integral_rules(eq, eqdata, dummyvars)
    phi, u, θ = dummyvars
    #! all that should be needed is to solve an integral problem, the trick is doing this
    #! with rules without putting symbols through the solve
end
