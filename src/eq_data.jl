struct EquationData <: PDEBase.AbstractVarEqMapping
    depvarmap
    indvarmap
    pde_indvars
    bc_indvars
    argmap
end

function EquationData(pdesys, v)
    eqs = pdesys.eqs
    bcs = pdesys.bcs
    alleqs = vcat(eqs, bcs)

    argmap = map(alleqs) do eq
        eq => get_argument([eq], v)[1]
    end
    depvarmap = map(alleqs) do eq
        eq => get_depvars(eq, v.depvar_ops)
    end
    indvarmap = map(alleqs) do eq
        eq => get_indvars(eq, indvars(v))
    end
    pde_indvars = if strategy isa QuadratureTraining
        get_argument(eqs, v)
    else
        get_variables(eqs, v)
    end

    bc_indvars = if strategy isa QuadratureTraining
        get_argument(bcs, v)
    else
        get_variables(bcs, v)
    end

    EquationData(depvarmap, indvarmap, pde_indvars, bc_depvars, argmap)
end

function depvars(eq, eqdata::EquationData)
    eqdata.depvarmap[eq]
end

function indvars(eq, eqdata::EquationData)
    eqdata.indvarmap[eq]
end

function pde_indvars(eqdata::EquationData)
    eqdata.pde_indvars
end

function bc_indvars(eqdata::EquationData)
    eqdata.bc_indvars
end

argument(eq, eqdata) = eqdata.argmap[eq]
