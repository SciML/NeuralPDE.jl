
function transform_inf_expr(dict_depvar_input, integrating_depvars, integrating_variables, transform)
    τs = Symbolics.variables(:τ, 1:length(integrating_variables))
    τs = Symbol.(τs)
    dict_transformation_vars = Dict()

    for depvar in integrating_depvars
        indvars = dict_depvar_input[depvar]
        ans = []

        for i in 1:length(indvars)
            if indvars[i] ∈ integrating_variables
                push!(ans, τs[i])
                dict_transformation_vars[indvars[i]] = transform(τs[i])
            else
                push!(ans, indvars[i])
            end
        end

        dict_depvar_input[depvar] = ans
    end

    this_eq_pair = Dict(map(intvars -> dict_depvars[intvars] => dict_depvar_input[intvars], integrating_depvars))
    this_eq_indvars = unique(vcat(values(this_eq_pair)...))

    return dict_transformation_vars, this_eq_indvars
end

function v_inf(t)
    return :($t ./ (1 .- $t.^2))
end

function v_semiinf(t , a , upto_inf)
    a = first(a)
    
    if upto_inf == true
        return :($a .+ $t ./ (1 .- $t))
    else
        return :($a .+ $t ./ (1 .+ $t))
    end
end

function get_inf_transformation_jacobian(integrating_variable, _inf, _semiup, _semilw)
    j = []
        for var in integrating_variable
            if _inf[1]
                append!(j, [:((1+$var^2)/(1-$var^2)^2)])
            elseif _semiup[1] || _semilw[1]
                append!(j, [:(1/(1-$var)^2)])
            end
        end

    return j
end

function transform_inf_integral(lb, ub, integrating_ex, integrating_depvars, dict_depvar_input, integrating_variable)
    if -Inf in lb || Inf in ub

        if !(integrating_variable isa Array)
            integrating_variable = [integrating_variable]
        end

        lbb = lb .== -Inf
        ubb = ub .== Inf
        _none = .!lbb .& .!ubb
        _inf = lbb .& ubb
        _semiup = .!lbb .& ubb
        _semilw = lbb  .& .!ubb

        lb = 0.00.*_semiup + -1.00.*_inf + -1.00.*_semilw +  _none.*lb
        ub = 1.00.*_semiup + 1.00.*_inf  + 0.00.*_semilw  + _none.*ub

        function transform_indvars(t)
            if _none[1]
                return t
            elseif _inf[1]
                return v_inf(t)
            elseif _semiup[1]
                return v_semiinf(t , lb , 1)
            elseif _semilw[1]
                return v_semiinf(t , ub , 0)
            end
        end

        @show dict_depvar_input[integrating_depvars[1]]
        @show integrating_depvars
        @show dict_depvar_input
        dict_transformation_vars, transformation_vars = transform_inf_expr(dict_depvar_input, integrating_depvars, integrating_variable,transform_indvars)

        j = get_inf_transformation_jacobian(integrating_variable, _inf, _semiup, _semilw)     
        
        integrating_ex = Expr(:call, :*, integrating_ex, j...)
    else
        dict_transformation_vars, transformation_vars = nothing, nothing
    end

    return lb, ub, integrating_ex, dict_transformation_vars, transformation_vars
end

