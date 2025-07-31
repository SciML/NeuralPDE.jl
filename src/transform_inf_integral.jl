
function transform_inf_expr(integrating_depvars, dict_depvar_input, dict_depvars,
        integrating_variables, transform)
    τs = Symbolics.variables(:τ, 1:length(integrating_variables))
    τs = Symbol.(τs)
    dict_transformation_vars = Dict()
    dict_depvar_input_ = Dict()
    integrating_var_transformation = []
    j = 1

    for depvar in integrating_depvars
        indvars = dict_depvar_input[depvar]
        ans = []

        for i in 1:length(indvars)
            if indvars[i] ∈ integrating_variables
                push!(ans, τs[j])
                push!(integrating_var_transformation, τs[j])
                dict_transformation_vars[indvars[i]] = transform(τs[j], j)
                j += 1
            else
                push!(ans, indvars[i])
            end
        end

        dict_depvar_input_[depvar] = ans
    end

    this_eq_pair = Dict(map(
        intvars -> dict_depvars[intvars] => dict_depvar_input_[intvars],
        integrating_depvars))
    this_eq_indvars = unique(vcat(values(this_eq_pair)...))

    return dict_transformation_vars, this_eq_indvars, integrating_var_transformation
end

function v_inf(t)
    return :($t ./ (1 .- $t .^ 2))
end

function v_semiinf(t, a, upto_inf)
    if a isa Num
        if upto_inf == true
            return :($t ./ (1 .- $t))
        else
            return :($t ./ (1 .+ $t))
        end
    end

    if upto_inf == true
        return :($a .+ $t ./ (1 .- $t))
    else
        return :($a .+ $t ./ (1 .+ $t))
    end
end

function get_inf_transformation_jacobian(integrating_variable, _inf, _semiup, _semilw,
        _num_semiup, _num_semilw)
    j = []
    for var in integrating_variable
        if _inf[1]
            append!(j, [:((1 + $var^2) / (1 - $var^2)^2)])
        elseif _semiup[1] || _num_semiup[1]
            append!(j, [:(1 / (1 - $var)^2)])
        elseif _semilw[1] || _num_semilw[1]
            append!(j, [:(1 / (1 + $var)^2)])
        end
    end

    return j
end

function transform_inf_integral(lb, ub, integrating_ex, integrating_depvars,
        dict_depvar_input, dict_depvars, integrating_variable,
        eltypeθ; dict_transformation_vars = nothing,
        transformation_vars = nothing)
    lb_ = Symbolics.tosymbol.(lb)
    ub_ = Symbolics.tosymbol.(ub)

    if -Inf in lb_ || Inf in ub_
        if !(integrating_variable isa Array)
            integrating_variable = [integrating_variable]
        end

        lbb = lb_ .=== -Inf
        ubb = ub_ .=== Inf
        _num_semiup = isa.(lb_, Symbol)
        _num_semilw = isa.(ub_, Symbol)
        _none = .!lbb .& .!ubb
        _inf = lbb .& ubb
        _semiup = .!lbb .& ubb .& .!_num_semiup
        _semilw = lbb .& .!ubb .& .!_num_semilw

        function transform_indvars(t, i)
            if _none[1]
                return t
            elseif _inf[1]
                return v_inf(t)
            elseif _semiup[1] || _num_semiup[1]
                return v_semiinf(t, lb[i], 1)
            elseif _semilw[1] || _num_semilw[1]
                return v_semiinf(t, ub[i], 0)
            end
        end

        dict_transformation_vars, transformation_vars,
        integrating_var_transformation = transform_inf_expr(
            integrating_depvars, dict_depvar_input, dict_depvars, integrating_variable, transform_indvars)

        ϵ = 1 / 20 #cbrt(eps(eltypeθ))

        lb = 0.00 .* _semiup + (-1.00 + ϵ) .* _inf + (-1.00 + ϵ) .* _semilw + _none .* lb +
             lb ./ (1 .+ lb) .* _num_semiup + (-1.00 + ϵ) .* _num_semilw
        ub = (1.00 - ϵ) .* _semiup + (1.00 - ϵ) .* _inf + 0.00 .* _semilw + _none .* ub +
             (1.00 - ϵ) .* _num_semiup + ub ./ (1 .+ ub) .* _num_semilw

        j = get_inf_transformation_jacobian(integrating_var_transformation, _inf, _semiup,
            _semilw, _num_semiup, _num_semilw)

        integrating_ex = Expr(:call, :*, integrating_ex, j...)
    end

    return lb, ub, integrating_ex, dict_transformation_vars, transformation_vars
end
