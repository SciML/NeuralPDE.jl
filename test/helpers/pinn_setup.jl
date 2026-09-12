# Shared setup for the PhysicsInformedNN test groups.
using NeuralPDE, ModelingToolkit, SciMLBase, Lux, Random, Test, Integrals, QuasiMonteCarlo
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using LineSearches: BackTracking
import DomainSets
import DomainSets: Interval, infimum, supremum

"""
Adam with the (optional, resampling) callback followed by BFGS on fixed collocation points.
"""
function train(prob; adam_iters = 500, bfgs_iters = 500, callback = (state, loss) -> false)
    res = solve(prob, Adam(0.01); maxiters = adam_iters, callback)
    prob = remake(prob; u0 = res.original_sol.u)
    return solve(prob, BFGS(; linesearch = BackTracking()); maxiters = bfgs_iters)
end

pde_strategies() = [
    GridTraining(0.1),
    StochasticTraining(200; bcs_points = 50),
    QuasiRandomTraining(100; sampling_alg = LatinHypercubeSample(), resampling = false),
    QuasiRandomTraining(100; bcs_points = 50, sampling_alg = LatticeRuleSample()),
    QuadratureTraining(; quadrature_alg = GaussLegendre(n = 10)),
]

"""
Callback drawing new collocation points every 25 iterations for resampling strategies.
"""
function resampling_callback(prob)
    md = pinn_metadata(prob)
    return (state, loss) -> (state.iter % 25 == 0 && resample!(state.p, md); false)
end
