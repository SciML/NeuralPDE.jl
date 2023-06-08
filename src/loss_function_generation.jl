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
        multioutput, init_params, strategy, eq_params,
        param_estim, default_p = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    eq = eq isa Equation ? eq.lhs : eq

    eq_args = get(eqdata.ivargs, eq, varmap.x̄)

    if integrand isa Nothing
        this_eq_indvars = indvars(eq, eqdata)
        this_eq_depvars = depvars(eq, eqdata)
        loss_function = parse_equation(pinnrep, eq, eq_iv_args(eq, eqdata))
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
        num_numbers = 0
        mapreduce(vcat, enumerate(eq_args)) do (i, x)
            if x isa Number
                num_numbers += 1
                fill(convert(eltypeθ, x), size(cord[[1], :]))
            else
                cord[[i-num_numbers], :]
            end
        end
    end

    full_loss_func = (cord, θ, phi, u, p) -> begin
        loss_function(get_coords(cord), θ, phi, u, get_ps(θ))
    end
    return full_loss_func
end

function build_loss_function(pinnrep, eqs)
     @unpack eq_params, param_estim, default_p, phi, derivative, integral = pinnrep

    _loss_function = build_symbolic_loss_function(pinnrep, eqs,
                                                      eq_params = eq_params,
                                                      param_estim = param_estim)
    u = get_u()
    loss_function = (cord, θ) -> begin _loss_function(cord, θ, phi, u,
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

function parse_equation(pinnrep::PINNRepresentation, term, ivs; is_integral = false,
                               dict_transformation_vars = nothing,
                               transformation_vars = nothing)
    @unpack varmap, eqdata, derivative, integral, flat_init_params = pinnrep
    eltypeθ = eltype(flat_init_params)

    ex_vars = get_depvars(term, varmap.depvar_ops)
    ignore = vcat(operation.(ex_vars), getindex, Differential, Integral, ~)

    dummyvars = @variables phi, u(..), θ_SYMBOL, coord
    dummyvars = unwrap.(dummyvars)
    deriv_rules = generate_derivative_rules(term, eqdata, eltypeθ, dummyvars, derivative, varmap)

    ch = Prewalk(Chain(deriv_rules))

    expr = ch(term)

    sym_coords = DestructuredArgs(ivs)
    ps = DestructuredArgs(varmap.ps)


    args = [coord, θ_SYMBOL, phi, u, ps]

    ex = Func(args, [], expr) |> toexpr |> _dot_
    @show ex
    f = @RuntimeGeneratedFunction ex
    return f
end

function generate_derivative_rules(term, eqdata, eltypeθ, dummyvars, derivative, varmap)
    phi, u, θ, coord = dummyvars
    dvs = depvars(term, eqdata)
    @show dvs
    # Orthodox derivatives
    rs = reduce(vcat, [reduce(vcat, [[@rule $((Differential(x)^d)(w)) =>
                                          derivative(phi,
                                                     u, coord,
                                                     [get_ε(length(arguments(w)),
                                                           j, eltypeθ, i) for i in 1:d],
                                                     d, θ)
            for d in differential_order(term, x)]
           for (j, x) in enumerate(varmap.args[operation(w)])], init = [])
          for w in dvs], init = [])
    # Mixed derivatives
    mx = mapreduce(vcat, dvs, init = []) do w
        mapreduce(vcat, enumerate(varmap.args[operation(w)]), init = []) do (j, x)
            mapreduce(vcat, enumerate(varmap.args[operation(w)]), init = []) do (k, y)
                if isequal(x, y)
                    (_) -> nothing
                else
                    n = length(arguments(w))
                    [@rule $((Differential(x))((Differential(y))(w))) =>
                        derivative(phi,
                                   (cord_, θ_, phi_) ->
                                       derivative(phi_, u, cord_,
                                                  [get_ϵ(n, k, eltypeθ, i) for i in 1:2], 1, θ_),
                                   coord, [get_ε(n, j, eltypeθ, 2)], 1, θ)]
                end
            end
        end
    end
    vr = mapreduce(vcat, dvs, init = []) do w
        @rule w => u(coord, θ, phi)
    end

    return [mx; rs; vr]
end

function generate_integral_rules(eq, eqdata, dummyvars)
    phi, u, θ = dummyvars
    #! all that should be needed is to solve an integral problem, the trick is doing this
    #! with rules without putting symbols through the solve

end
