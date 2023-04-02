using Optimization, OptimizationOptimisers
using Test, NeuralPDE
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
using Plots, Revise
import Lux

nonadaptive_loss = NeuralPDE.NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)
gradnormadaptive_loss = NeuralPDE.GradientScaleAdaptiveLoss(100, pde_loss_weights = 1e3,
                                                            bc_loss_weights = 1)
minimaxadaptive_loss = NeuralPDE.MiniMaxAdaptiveLoss(100; pde_loss_weights = 1,
                                              bc_loss_weights = 1)
neuraltangentkernel_loss = NeuralPDE.NeuralTangentKernelAdaptiveLoss(100; pde_loss_weights = 1,
                                         bc_loss_weights = 1)
inversedirichlet_loss = NeuralPDE.InverseDirichletAdaptiveLoss(100, pde_loss_weights = 1e3,
                                        bc_loss_weights = 1)
adaptive_losses = [nonadaptive_loss, gradnormadaptive_loss, minimaxadaptive_loss, neuraltangentkernel_loss, inversedirichlet_loss]
maxiters = 4000
seed = 60

## 2D Poisson equation
function test_2d_poisson_equation_adaptive_loss(adaptive_loss; seed = 60, maxiters = 4000)
    Random.seed!(seed)
    hid = 40
    chain_ = Lux.Chain(Lux.Dense(2, hid, Lux.σ), Lux.Dense(hid, hid, Lux.σ),
                       Lux.Dense(hid, 1))
    strategy_ = NeuralPDE.StochasticTraining(256)
    @info "adaptive reweighting test outdir:, maxiters: $(maxiters), 2D Poisson equation, adaptive_loss: $(nameof(typeof(adaptive_loss))) "
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    # Initial and boundary conditions
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
        u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    iteration = [0]
    discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                 strategy_;
                                                 adaptive_loss = adaptive_loss,
                                                 logger = nothing,
                                                 iteration = iteration)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = NeuralPDE.discretize(pde_system, discretization)
    phi = discretization.phi
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)
    u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                     (length(xs), length(ys)))

    callback = function (p, l)
        iteration[1] += 1
        if iteration[1] % 100 == 0
            @info "Current loss is: $l, iteration is $(iteration[1])"
        end
        return false
    end
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.03); maxiters = maxiters,
                             callback = callback)

    u_predict = reshape([first(phi([x, y], res.minimizer)) for x in xs for y in ys],
                        (length(xs), length(ys)))
    diff_u = abs.(u_predict .- u_real)
    total_diff = sum(diff_u)
    total_u = sum(abs.(u_real))
    total_diff_rel = total_diff / total_u

    #p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    #p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    #p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    #(plot=plot(p1,p2,p3), error=total_diff, total_diff_rel=total_diff_rel)
    (error = total_diff, total_diff_rel = total_diff_rel)
end

@info "testing that the adaptive loss methods roughly succeed"
function test_2d_poisson_equation_adaptive_loss_no_logs_run_seediters(adaptive_loss)
    test_2d_poisson_equation_adaptive_loss(adaptive_loss; seed = seed, maxiters = maxiters)
end
error_results_no_logs = map(test_2d_poisson_equation_adaptive_loss_no_logs_run_seediters,
                            adaptive_losses)

# accuracy tests
@show error_results_no_logs[1][:total_diff_rel]
@show error_results_no_logs[2][:total_diff_rel]
@show error_results_no_logs[3][:total_diff_rel]
# accuracy tests, these work for this specific seed but might not for others
# note that this doesn't test that the adaptive losses are outperforming the nonadaptive loss, which is not guaranteed, and seed/arch/hyperparam/pde etc dependent
@test error_results_no_logs[1][:total_diff_rel] < 0.4
@test error_results_no_logs[2][:total_diff_rel] < 0.4
@test error_results_no_logs[3][:total_diff_rel] < 0.4

#plots_diffs[1][:plot]
#plots_diffs[2][:plot]
#plots_diffs[3][:plot]
