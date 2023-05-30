# TODO: add multioutput
# TODO: add integrals

function build_symbolic_loss_function(pinnrep::PINNRepresentation, eq;
                                      eq_params = SciMLBase.NullParameters(),
                                      param_estim = false,
                                      default_p = [],
                                      integrand = nothing,
                                      transformation_vars = nothing)
    @unpack v, eqdata,
        phi, derivative, integral,
        multioutput, init_params, strategy, eq_params,
        param_estim, default_p = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    eq_args = get(eqdata.eq_args, eq, v.x̄)

    if integrand isa Nothing
        this_eq_indvars = indvars(eq, eqmap)
        this_eq_depvars = depvars(eq, eqmap)
        loss_function = parse_equation(pinnrep, eq, eq_iv_args(eq, eqmap))
    else
        this_eq_indvars = transformation_vars isa Nothing ?
                          unique(indvars(eq, eqmap)) : transformation_vars
        loss_function = integrand
    end

    n = length(this_eq_indvars)

    if param_estim == true && eq_params != SciMLBase.NullParameters()
        param_len = length(eq_params)
        # check parameter format to use correct indexing
        psform = (phi isa Vector && phi[1].f isa Optimisers.Restructure) ||
        (!(phi isa Vector) && phi.f isa Optimisers.Restructure)

        if psform
            last_indx = [0; accumulate(+, map(length, init_params))][end]
            ps_range = 1:param_len .+ last_indx
            get_ps = (θ) -> θ[ps_range]
        else
            ps_range = 1:param_len
            get_ps = (θ) -> θ.p[ps_range]
        end
    else
        get_ps = (θ) -> default_p
    end

    function get_coords(cord)
        map(enumerate(eq_args)) do (i, x)
            if x isa Number
                fill(x, size(cord[[1], :]))
            else
                cord[[i], :]
            end
        end
    end

    full_loss_func = (cord, θ, phi, derivative, integral, u, p) -> begin
        loss_function(get_coords(cord), θ, phi, derivative, integral, u, get_ps(θ))
    end
    return full_loss_func
end

function build_loss_function(pinnrep, eqs)
     @unpack eq_params, param_estim, default_p, phi, derivative, integral = pinnrep

    _loss_function = build_symbolic_loss_function(pinnrep, eqs,
                                                      eq_params = eq_params,
                                                      param_estim = param_estim)

    u = get_u()
    loss_function = (cord, θ) -> begin _loss_function(cord, θ, phi, derivative, integral, u,
                                                      default_p) end
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

function parse_equation(pinnrep::PINNRepresentation, eq, ivs; is_integral = false,
                               dict_transformation_vars = nothing,
                               transformation_vars = nothing)
    @unpack v, eqdata, derivative, integral = pinnrep

    expr = eq isa Equation ? eq.lhs : eq
    ex_vars = vars(expr)
    ignore = vcat(operation.(ex_vars), getindex, Differential, Integral, ~)
    ex_ops = filter(x -> !any(isequal(x), ignore), ex_ops)
    op_rules = [@rule $(op)(~~a) => broadcast(op, ~a...) for op in ex_ops]

    dummyvars = @variables phi, u, θ
    deriv_rules = generate_derivative_rules(eq, eqdata, dummyvars)

    ch = Postwalk(Chain([deriv_rules; op_rules]))
    expr = ch(expr)

    sym_coords = DestructuredArgs(ivs)
    ps = DestructuredArgs(v.ps)


    args = [sym_coords, θ, phi, u, ps]

    ex = Func(args, [], expr) |> toexpr

    return ex
end

function generate_derivative_rules(eq, eqdata, dummyvars)
    phi, u, θ = dummyvars
    rs = [@rule $(Differential(~x)^(~d::isinteger)(~w)) => derivative(phi, u, ~x, get_εs(~w), ~d, θ)]
    # TODO: add mixed derivatives
    return rs
end

function generate_integral_rules(eq, eqdata, dummyvars)
    phi, u, θ = dummyvars
    #! all that should be needed is to solve an integral problem, the trick is doing this
    #! with rules without putting symbols through the solve

end
