# TODO: add multioutput
# TODO: add integrals

function build_symbolic_loss_function(pinnrep::PINNRepresentation, eq;
                                      eq_params = SciMLBase.NullParameters(),
                                      param_estim = false,
                                      default_p = [],
                                      integrand = nothing,
                                      transformation_vars = nothing)
    @unpack varmap, eqdata,
        phi, phimap, derivative, integral,
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
            @show length(phi)
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
        out = map(enumerate(eq_args)) do (i, x)
            if x isa Number
                num_numbers += 1
                fill(convert(eltypeθ, x), length(cord[[1], :]))
            else
                cord[[i-num_numbers], :]
            end
        end
        if out === nothing
            return []
        else
            return out
        end
    end

    full_loss_func = (cord, θ, phi, p) -> begin
        coords = get_coords(cord)
        @show coords
        combinedcoords = reduce(vcat, coords, init = [])
        @show combinedcoords
        loss_function(coords, combinedcoords, θ, phi, get_ps(θ))
    end
    return full_loss_func
end

function build_loss_function(pinnrep, eqs)
     @unpack eq_params, param_estim, default_p, phi, phimap, multioutput, derivative, integral = pinnrep

    if multioutput
       phi = phimap
    end

    _loss_function = build_symbolic_loss_function(pinnrep, eqs,
                                                      eq_params = eq_params,
                                                      param_estim = param_estim)
    loss_function = (cord, θ) -> begin _loss_function(cord, θ, phi,
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

    dummyvars = @variables phi(..), θ_SYMBOL, coord
    dummyvars = unwrap.(dummyvars)
    deriv_rules = generate_derivative_rules(term, eqdata, eltypeθ, dummyvars, derivative, varmap)

    ch = Prewalk(Chain(deriv_rules))

    expr = ch(term)

    sym_coords = DestructuredArgs(ivs)
    ps = DestructuredArgs(varmap.ps)


    args = [sym_coords, coord, θ_SYMBOL, phi, ps]

    ex = Func(args, [], expr) |> toexpr |> _dot_
    f = @RuntimeGeneratedFunction ex
    return f
end

function generate_derivative_rules(term, eqdata, eltypeθ, dummyvars, derivative, varmap)
    phi, θ, coord = dummyvars
    dvs = get_depvars(term, varmap.depvar_ops)
    @show dvs
    # Orthodox derivatives
    n(w) = length(arguments(w))
    rs = reduce(vcat, [reduce(vcat, [[@rule $((Differential(x)^d)(w)) =>
                                          derivative(ufunc(w, coord, θ, phi), coord,
                                                     [get_ε(n(w),
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
                    [(_) -> nothing]
                else
                    ε1 = [get_ε(n(w), j, eltypeθ, i) for i in 1:2]
                    ε2 = [get_ε(n(w), k, eltypeθ, i) for i in 1:2]
                    [@rule $((Differential(x))((Differential(y))(w))) =>
                        derivative((cord_, θ_) -> derivative(ufunc(w, coord, θ, phi), cord_,
                                                             ε2, 1, θ_),
                                   coord, ε1, 1, θ)]
                end
            end
        end
    end
    vr = mapreduce(vcat, dvs, init = []) do w
        @rule w => ufunc(w, coord, θ, phi)(coord, θ)
    end

    return [mx; rs; vr]
end

function generate_integral_rules(eq, eqdata, dummyvars)
    phi, u, θ = dummyvars
    #! all that should be needed is to solve an integral problem, the trick is doing this
    #! with rules without putting symbols through the solve

end
