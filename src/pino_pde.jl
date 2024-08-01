
#TODO?
function PhysicsInformedNO(
        neural_operator,
        parameters,#bounds
        strategy;
        kwargs...
    )
    PhysicsInformedNN(neural_operator,
        strategy; kwargs...
    )
end
#TODO?
function SciMLBase.discretize(pde_system::PDESystem, neural_operator::PhysicsInformedNO)
    pinnrep = symbolic_discretize(pde_system, neural_operator)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
        Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
#TODO?
function SciMLBase.discretize(pde_system::PDESystem, neural_operator::PhysicsInformed)
    pinnrep = symbolic_discretize(pde_system, neural_operator)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
        Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
