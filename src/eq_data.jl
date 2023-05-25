struct EquationData <: PDEBase.AbstractVarEqMapping
    depvarmap
    indvarmap
    args
    ivargs
    argmap
end

function EquationData(pdesys, v)
    eqs = pdesys.eqs
    bcs = pdesys.bcs
    alleqs = vcat(eqs, bcs)

    argmap = map(alleqs) do eq
        eq => get_argument([eq], v)[1]
    end |> Dict
    depvarmap = map(alleqs) do eq
        eq => get_depvars(eq, v.depvar_ops)
    end |> Dict
    indvarmap = map(alleqs) do eq
        eq => get_indvars(eq, indvars(v))
    end |> Dict

    args = map(alleqs) do eq
        if strategy isa QuadratureTraining
            eq => get_argument(bcs, v)
        else
            eq => get_variables(bcs, v)
        end
    end |> Dict

    ivargs = map(alleqs) do eq
        if strategy isa QuadratureTraining
            eq => get_iv_argument(eqs, v)
        else
            eq => get_iv_variables(eqs, v)
        end
    end |> Dict

    EquationData(depvarmap, indvarmap, args, ivargs, argmap)
end

function depvars(eq, eqdata::EquationData)
    eqdata.depvarmap[eq]
end

function indvars(eq, eqdata::EquationData)
    eqdata.indvarmap[eq]
end

function eq_args(eq, eqdata::EquationData)
    eqdata.args[eq]
end

function eq_iv_args(eq, eqdata::EquationData)
    eqdata.ivargs[eq]
end

argument(eq, eqdata) = eqdata.argmap[eq]


function get_iv_argument(eqs, v::VariableMap)
    vars = map(eqs) do eq
        _vars = map(depvar -> get_depvars(eq, depvar), v.depvar_ops)
        f_vars = filter(x -> !isempty(x), _vars)
        v.args[operation(map(x -> first(x), f_vars))]
    end
    args_ = map(vars) do _vars
        seen = []
            args_ = map(vars) do _vars
        seen = []
        filter(reduce(vcat, arguments.(_vars))) do x
            if x isa Number
                true
            else
                if any(isequal(x), seen)
                    false
                else
                    push!(seen, x)
                    true
                end
            end
        end
    end
    return args_ # TODO for all arguments
end

"""
``julia
get_variables(eqs,_indvars,_depvars)
```

Returns all variables that are used in each equations or boundary condition.
"""
function get_iv_variables(eqs, v::VariableMap)
    args = get_iv_argument(eqs, v)
    return map(arg -> filter(x -> !(x isa Number), arg), args)
end
