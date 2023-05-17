# TODO: add multioutput
# TODO: add param_estim
# TODO: add integrals

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
        this_eq_indvars = indvars(eq, eqmap)
        this_eq_depvars = depvars(eq, eqmap)
        loss_function = parse_equation(pinnrep, eq, this_eq_indvars)
    else
        this_eq_indvars = transformation_vars isa Nothing ?
                          unique(indvars(eq, eqmap)) : transformation_vars
        loss_function = integrand
    end

    n = length(this_eq_indvars)

    full_loss_func = (cord, θ, phi, derivative, integral, u, p) -> begin
        ivs = [cord[[i], :] for i in 1:n]
        cords = map(this_eq_depvars) do w
            idxs = map(x -> x2i(v, w, x), v.args[operation(w)])
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

function parse_equation(pinnrep::PINNRepresentation, eq, ivs; is_integral = false,
                               dict_transformation_vars = nothing,
                               transformation_vars = nothing)
    @unpack v, eqdata, derivative, integral = pinnrep

    expr = eq.lhs
    ex_vars = vars(expr)
    ignore = vcat(operation.(ex_vars), getindex, Differential, Integral, ~)
    ex_ops = filter(x -> !any(isequal(x), ignore), ex_ops)
    op_rules = [@rule $(op)(~~a) => broadcast(op, ~a...) for op in ex_ops]

    dummyvars = @variables phi, u, θ
    deriv_rules = generate_derivative_rules(eq, eqdata, dummyvars)

    ch = Postwalk(Chain([deriv_rules; op_rules]))

    sym_coords = DestructuredArgs(ivs)

    expr = ch(expr)

    args = [phi, u, sym_coords, θ]

    ex = Func(args, [], expr) |> toexpr

    return ex
end

function generate_derivative_rules(eq, eqdata, dummyvars)
    phi, u, θ = dummyvars
    @register_symbolic derivative(phi, u, coord, εs, order, θ)
    rs = [@rule $(Differential(~x)^(~d)(~w)) => derivative(phi, u, ~x, get_εs(~w), ~d, θ)]
    # TODO: add mixed derivatives
    return rs
end
