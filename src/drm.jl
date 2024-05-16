struct DeepRitz <: AbstractPINN
    PINN::PhysicsInformedNN
end

DeepRitz(chain, strategy::T; kwargs...) where T = DeepRitz(PhysicsInformedNN(chain, strategy; kwargs...))

"""
    prob = discretize(pde_system::PDESystem, discretization::DeepRitz)

Transforms a symbolic description of a ModelingToolkit-defined `PDESystem` and generates
an `OptimizationProblem` for [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) whose
solution is the solution to the PDE.
"""
function SciMLBase.discretize(pde_system::PDESystem, discretization::DeepRitz)
    modify_deep_ritz!(pde_system);
    pinnrep = symbolic_discretize(pde_system, discretization)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
        Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end



function modify_deep_ritz!(pde_system::PDESystem)

    if length(pde_system.eqs) > 1
        error("Deep Ritz solves for only one dependent variable")
    end

    ind_vars = pde_system.ivs
    dep_var = pde_system.dvs[1]

    expr = first(pde_system.eqs).lhs - first(pde_system.eqs).rhs

    Ds = [Differential(ind_var) for ind_var in ind_vars];
    D²s = [Differential(ind_var)^2 for ind_var in ind_vars];
    laplacian = (sum([d²s(dep_var) for d²s in D²s]) ~ 0).lhs;
    

    # Put checks here and get the correct rhs. For now, let's go with Δu ~ f(x) 

    rhs = (expr - laplacian) * dep_var
    lhs = (sum([(ds(dep_var))^2 for ds in Ds]) ~ 0).lhs;  # check this for potential bugs

    pde_system.eqs[1] = lhs ~ rhs
end
