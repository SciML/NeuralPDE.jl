
function transform_inf_expr(integrating_depvars, dict_depvar_input, dict_depvars, integrating_variables, transform)
    τs = Symbolics.variables(:τ, 1:length(integrating_variables))
    τs = Symbol.(τs)
    dict_transformation_vars = Dict()
    dict_depvar_input_ = Dict()
    integrating_var_transformation = []

    for depvar in integrating_depvars
        indvars = dict_depvar_input[depvar]
        ans = []

        for i in 1:length(indvars)
            if indvars[i] ∈ integrating_variables
                push!(ans, τs[i])
                push!(integrating_var_transformation, τs[i])
                dict_transformation_vars[indvars[i]] = transform(τs[i])
            else
                push!(ans, indvars[i])
            end
        end

        dict_depvar_input_[depvar] = ans
    end

    this_eq_pair = Dict(map(intvars -> dict_depvars[intvars] => dict_depvar_input_[intvars], integrating_depvars))
    this_eq_indvars = unique(vcat(values(this_eq_pair)...))

    return dict_transformation_vars, this_eq_indvars,integrating_var_transformation
end

function v_inf(t)
    return :($t ./ (1 .- $t.^2))
end

function v_semiinf(t , a , upto_inf, indvars, depvars, 
                   dict_indvars, dict_depvars, dict_depvar_input, 
                   phi, derivative_, chain, initθ, strategy, integrating_depvars)
   
    a = first(a)

    if a isa Num 
        #=
        a_f = NeuralPDE.build_symbolic_loss_function(nothing, indvars,depvars,
                                                            dict_indvars,dict_depvars,
                                                            dict_depvar_input, phi, derivative_,
                                                            nothing, chain, initθ, strategy,
                                                            integrand = ex, integrating_depvars=integrating_depvars,
                                                            param_estim =false, default_p = nothing)
        a = @RuntimeGeneratedFunction(a_f)
        =#
        if upto_inf == true
            ex = :($a .+ $t ./ (1 .- $t))
        else
            ex = :($a .+ $t ./ (1 .+ $t))
        end
        
        a_f = NeuralPDE.build_symbolic_loss_function(nothing, indvars,depvars,
                                                            dict_indvars,dict_depvars,
                                                            dict_depvar_input, phi, derivative_,
                                                            nothing, chain, initθ, strategy,
                                                            integrand = ex, integrating_depvars=integrating_depvars,
                                                            param_estim =false, default_p = nothing)
        return @RuntimeGeneratedFunction(a_f)
    end 
    
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

function transform_inf_integral(lb, ub, integrating_variable, integrating_ex,
                                indvars, depvars, dict_indvars, dict_depvars, 
                                dict_depvar_input, phi, derivative_, chain, 
                                initθ, strategy, integrating_depvars; 
                                dict_transformation_vars = nothing, transformation_vars = nothing)

    lb_ = Symbolics.tosymbol.(lb)
    ub_ = Symbolics.tosymbol.(ub)

    if -Inf in lb_ || Inf in ub_

        if !(integrating_variable isa Array)
            integrating_variable = [integrating_variable]
        end

        lbb = lb_ .=== -Inf
        ubb = ub_ .=== Inf
        _none = .!lbb .& .!ubb
        _inf = lbb .& ubb
        _semiup = .!lbb .& ubb
        _semilw = lbb  .& .!ubb

        function transform_indvars(t)
            if _none[1]
                return t
            elseif _inf[1]
                return v_inf(t)
            elseif _semiup[1]
                return v_semiinf(t , lb , 1, indvars, depvars, 
                                 dict_indvars, dict_depvars, dict_depvar_input, 
                                 phi, derivative_, chain, initθ, strategy, integrating_depvars)
            elseif _semilw[1]
                return _semiinf(t , ub , 0, indvars, depvars, 
                                dict_indvars, dict_depvars, dict_depvar_input, 
                                phi, derivative_, chain, initθ, strategy, integrating_depvars)
            end
        end

        dict_transformation_vars, transformation_vars, integrating_var_transformation = transform_inf_expr(integrating_depvars, dict_depvar_input, dict_depvars, integrating_variable,transform_indvars)

        ex = :(x .+ τ₁ ./ (1 .+ τ₁))
        
        a_f = NeuralPDE.build_symbolic_loss_function(nothing, indvars,depvars,
                                                            dict_indvars,dict_depvars,
                                                            dict_depvar_input, phi, derivative_,
                                                            nothing, chain, initθ, strategy,
                                                            integrand = ex, integrating_depvars=integrating_depvars,
                                                            param_estim =false, default_p = nothing)

        dict_transformation_vars = Dict(:x2 => @RuntimeGeneratedFunction(a_f))

        ϵ = 1/20 #cbrt(eps(eltypeθ))

        lb = 0.00.*_semiup + (-1.00+ϵ).*_inf + (-1.00+ϵ).*_semilw +  _none.*lb
        ub = (1.00-ϵ).*_semiup + (1.00-ϵ).*_inf  + 0.00.*_semilw  + _none.*ub

        j = get_inf_transformation_jacobian(integrating_var_transformation, _inf, _semiup, _semilw)     

        integrating_ex = Expr(:call, :*, integrating_ex, j...)
    end

    return lb, ub, integrating_ex, dict_transformation_vars, transformation_vars
end

println("Infinity Integral Test")
@parameters x
@variables u(..)
I = Integral(x in ClosedInterval(1, x))
Iinf = Integral(x in ClosedInterval(x, Inf))
eqs = [Iinf(u(x)) ~ - 1/x]
bcs = [u(1) ~ 1]
domains = [x ∈ Interval(1.0, 2.0)]
chain = FastChain(FastDense(1, 10, Flux.σ), FastDense(10, 1))
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
discretization = NeuralPDE.PhysicsInformedNN(chain, NeuralPDE.GridTraining(0.1), init_params= initθ)
@named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
sym_prob = SciMLBase.symbolic_discretize(pde_system, discretization)
@show sym_prob
prob = SciMLBase.discretize(pde_system, discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=300)



sym_prob = (Expr[:((cord, var"##θ#257", phi, derivative, integral, u, p)->begin
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:600 =#
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:600 =#
          begin
              let (x,) = (cord[[1], :],)
                  begin
                      cord1 = vcat(x)
                  end
                  integral(u, cord1, phi, [1], RuntimeGeneratedFunctions.RuntimeGeneratedFunction{(:cord, Symbol("##θ#257"), :phi, :derivative, :integral, :u, :p), NeuralPDE.var"#_RGF_ModTag", NeuralPDE.var"#_RGF_ModTag", (0xbb407138, 0x9a2ceadd, 0x705a3080, 0x671ac0d8, 0xa3b6b0c6)}(quote
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:600 =#
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:600 =#
    begin
        let (τ₁,) = (cord[[1], :],)
            begin
                x2 = x .+ τ₁ ./ (1 .- τ₁)
            end
            begin
                cord1 = vcat(x2)
            end
            (*).(u(cord1, var"##θ#257", phi), (/).(1, (^).((-).(1, τ₁), 2)))
        end
    end
end), Any[0.0], Any[0.95], var"##θ#257")
...