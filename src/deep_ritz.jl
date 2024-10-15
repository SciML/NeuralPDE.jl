"""
    DeepRitz(chain,
                strategy;
                init_params = nothing,
                phi = nothing,
                param_estim = false,
                additional_loss = nothing,
                adaptive_loss = nothing,
                logger = nothing,
                log_options = LogOptions(),
                iteration = nothing,
                kwargs...)

A `discretize` algorithm for the ModelingToolkit PDESystem interface, which transforms a
`PDESystem` into an `OptimizationProblem` for the Deep Ritz method.

## Positional Arguments

* `chain`: a vector of Lux/Flux chains with a d-dimensional input and a
           1-dimensional output corresponding to each of the dependent variables. Note that this
           specification respects the order of the dependent variables as specified in the PDESystem.
           Flux chains will be converted to Lux internally using `adapt(FromFluxAdaptor(false, false), chain)`.
* `strategy`: determines which training strategy will be used. See the Training Strategy
              documentation for more details.

## Keyword Arguments

* `init_params`: the initial parameters of the neural networks. If `init_params` is not
  given, then the neural network default parameters are used. Note that for Lux, the default
  will convert to Float64.
* `phi`: a trial solution, specified as `phi(x,p)` where `x` is the coordinates vector for
  the dependent variable and `p` are the weights of the phi function (generally the weights
  of the neural network defining `phi`). By default, this is generated from the `chain`. This
  should only be used to more directly impose functional information in the training problem,
  for example imposing the boundary condition by the test function formulation.
* `adaptive_loss`: the choice for the adaptive loss function. See the
  [adaptive loss page](@ref adaptive_loss) for more details. Defaults to no adaptivity.
* `additional_loss`: a function `additional_loss(phi, θ, p_)` where `phi` are the neural
  network trial solutions, `θ` are the weights of the neural network(s), and `p_` are the
  hyperparameters of the `OptimizationProblem`. If `param_estim = true`, then `θ` additionally
  contains the parameters of the differential equation appended to the end of the vector.
* `param_estim`: whether the parameters of the differential equation should be included in
  the values sent to the `additional_loss` function. Defaults to `false`.
* `logger`: ?? needs docs
* `log_options`: ?? why is this separate from the logger?
* `iteration`: used to control the iteration counter???
* `
"""
struct DeepRitz{T, P, PH, DER, PE, AL, ADA, LOG, K} <: AbstractPINN
    chain::Any
    strategy::T
    init_params::P
    phi::PH
    derivative::DER
    param_estim::PE
    additional_loss::AL
    adaptive_loss::ADA
    logger::LOG
    log_options::LogOptions
    iteration::Vector{Int64}
    self_increment::Bool
    multioutput::Bool
    kwargs::K
end

function DeepRitz(chain, strategy; kwargs...)
    pinn = NeuralPDE.PhysicsInformedNN(chain, strategy);

    DeepRitz([
        getfield(pinn, k) for k in propertynames(pinn)]...)
end

"""
    prob = discretize(pde_system::PDESystem, discretization::DeepRitz)

For 2nd order PDEs, transforms a symbolic description of a ModelingToolkit-defined `PDESystem` 
using Deep-Ritz me and generates an `OptimizationProblem` for [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)  
whose solution is the solution to the PDE.
"""
function SciMLBase.discretize(pde_system::PDESystem, discretization::DeepRitz)
    modify_deep_ritz!(pde_system);
    pinnrep = symbolic_discretize(pde_system, discretization)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
        Optimization.AutoZygote())
    Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end


"""
    modify_deep_ritz!(pde_system::PDESystem)

Performs the checks for Deep-Ritz method and modifies the pde in the `pde_system`.
"""
function modify_deep_ritz!(pde_system::PDESystem)

    if length(pde_system.eqs) > 1
        error("Deep Ritz solves for only one dependent variable")
    end

    ind_vars = pde_system.ivs
    dep_var = pde_system.dvs[1]

    n_vars = length(ind_vars)

    expr = first(pde_system.eqs).lhs - first(pde_system.eqs).rhs

    Ds = [Differential(ind_var) for ind_var in ind_vars];
    D²s = [Differential(ind_var)^2 for ind_var in ind_vars];
    laplacian = (sum([d²s(dep_var) for d²s in D²s]) ~ 0).lhs;

    expr_new = modify_laplacian(expr, laplacian, n_vars);

    rhs = - expr_new * dep_var
    lhs = (sum([(ds(dep_var))^2 for ds in Ds]) ~ 0).lhs;

    pde_system.eqs[1] = lhs ~ rhs
    return nothing
end


function modify_laplacian(expr, Δ, n_vars)
    expr_new = expr - Δ;
    if (operation(expr_new)!= +) || (length(expr_new.dict) + n_vars == length(expr.dict))
        # positive coeff of laplacian
        return expr_new
    else
        expr_new = expr + Δ
        if length(expr_new.dict) == n_vars + length(expr.dict)
            # negative coeff of laplacian
            return expr_new
        else
            error("Incorrect form of PDE given")
        end
    end
end