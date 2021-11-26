
function transform_inf_expr(args, integrating_variables, transform; ans = [])
    for arg in args
        if arg ∈ integrating_variables
            push!(ans, transform(arg))
        else
            push!(ans, arg)
        end
    end
    return ans
end

function v_inf(t)
    return :($t ./ (1 .- $t.^2))
end

function v_semiinf(t , a , upto_inf)
    if upto_inf == true
        return :(a .+ $t ./ (1 .- $t))
    else
        return :(a .+ $t ./ (1 .+ $t))
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

function transform_inf_integral(lb, ub, integrating_ex, bc_indvars, integrating_variable)
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

        bc_indvars = transform_inf_expr(bc_indvars, integrating_variable,transform_indvars)

        j = get_inf_transformation_jacobian(integrating_variable, _inf, _semiup, _semilw)     
        
        integrating_ex = Expr(:call, :*, integrating_ex, j...)
    end

    return lb, ub, integrating_ex, bc_indvars
end
