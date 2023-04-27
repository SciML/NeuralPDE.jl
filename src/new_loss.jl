function build_symbolic_loss_function(pinnrep::PINNRepresentation, eq;
                                      eq_params = SciMLBase.NullParameters(),
                                      param_estim = false,
                                      default_p = nothing,
                                      bc_indvars = pinnrep.v.x̄,
                                      integrand = nothing,
                                      dict_transformation_vars = nothing,
                                      transformation_vars = nothing,
                                      integrating_depvars = pinnrep.v.ū)
    @unpack v, eqdata,
    phi, derivative, integral,
    multioutput, init_params, strategy, eq_params,
    param_estim, default_p = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    if integrand isa Nothing
        loss_function = parse_equation(pinnrep, eq)
        this_eq_indvars = indvars(eq, eqmap)
        this_eq_depvars = depvars(eq, eqmap)
    else
        this_eq_indvars = transformation_vars isa Nothing ?
                          unique(indvars(eq, eqmap)) : transformation_vars
        loss_function = integrand
    end

    n = length(this_eq_indvars)

    full_loss_func = (cord, θ, phi, derivative, integral, u, p) -> begin
        ivs = [cord[[i], :] for i in 1:n]
        cords = map(this_eq_depvars) do w
            idxs = map(x -> x2i(v, w, x), v.args[operation(w)]))
            vcat(ivs[idxs]...)
        end
        loss_function(cords, θ, phi, derivative, integral, u, p)
    end
end

function operations(ex)
    if istree(ex)
        op = operation(ex)
        return vcat(operations.(arguments(ex))..., op)
    end
    return []
end

function parse_equation(pinnrep::PINNRepresentation, ex; is_integral = false,
                               dict_transformation_vars = nothing,
                               transformation_vars = nothing)
    @unpack v, eqdata, derivative, integral = pinnrep

    expr = scalarize(ex)
    ex_vars = vars(expr)
    ignore = vcat(operation.(ex_vars), getindex, Differential, Integral, ~)
    ex_ops = filter(x -> !any(isequal(x), ignore), ex_ops)
    op_rules = [@rule $(op)(~~a) => broadcast(op, ~a...) for op in ex_ops]

    dummyvars = @variables phi, u, x, θ
    deriv_rules = generate_derivatives_rules(eq, eqdata, dummyvars)

    ch = Postwalk(Chain([deriv_rules; op_rules]))
    expr = ch(expr)

    args = [phi, u, x, θ]

    ex = Func(args, [], eq.rhs) |> toexpr


end

function generate_derivative_rules(eq, eqdata, dummyvars)
    phi, u, coord, θ = dummyvars
    @register_symbolic derivative(phi, u, coord, εs, order, θ)
    rs = [[@rule $(Differential(x)^(~d)(w)) => derivative(phi, u, coord, get_εs(w), d, θ)
           for x in all_ivs(w, v)]
          for w in depvars(eq, eqdata)]
    # TODO: add mixed derivatives
    return reduce(vcat, rs)
end
