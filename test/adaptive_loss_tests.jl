println("adaptive_reweighting_tests")
using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
#using Plots
println("Starting Soon!")

nonadaptive_loss = NeuralPDE.NonAdaptiveLoss(pde_loss_weights=1, bc_loss_weights=1)
gradnormadaptive_loss = NeuralPDE.GradientScaleAdaptiveLoss(100, pde_loss_weights=1e3, bc_loss_weights=1)
adaptive_loss = NeuralPDE.MiniMaxAdaptiveLoss(100; pde_loss_weights=1, bc_loss_weights=1)
adaptive_losses = [nonadaptive_loss, gradnormadaptive_loss,adaptive_loss]
maxiters_logs=4000
maxiters_no_logs=200
seed=60

## 2D Poisson equation
function test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, outdir, haslogger; seed=60, maxiters=4000)
    println
    if !isdir(outdir)
        mkpath(outdir)
    end
    for dir in readdir(outdir)
        fullpath = joinpath(outdir, dir)
        @show "deleting $fullpath"
        rm(fullpath, recursive=true)
    end
    if haslogger
        loggerloc = joinpath(outdir, "$run")
        logger = TBLogger(loggerloc, tb_append) #create tensorboard logger
    else
        logger = nothing
    end
    Random.seed!(seed)
    hid = 40
    chain_ = FastChain(FastDense(2,hid,Flux.σ),FastDense(hid,hid,Flux.σ),FastDense(hid,1))
    strategy_ =  NeuralPDE.StochasticTraining(256)
    println("adaptive reweighting test outdir: $(outdir), maxiters: $(maxiters), 2D Poisson equation, adaptive_loss: $(nameof(typeof(adaptive_loss))) ")
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    # Initial and boundary conditions
    bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
           u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
    # Space and time domains
    domains = [x ∈ Interval(0.0,1.0),
               y ∈ Interval(0.0,1.0)]

    initθ = Float64.(DiffEqFlux.initial_params(chain_))
    iteration = [0]
    discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                 strategy_;
                                                 init_params = initθ,
                                                 adaptive_loss = adaptive_loss,
                                                 logger = logger,
                                                 iteration=iteration)


    @named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
    prob = NeuralPDE.discretize(pde_system,discretization)
    phi = discretization.phi
    sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)


    xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)
    u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))

    cb = function (p,l)
        iteration[1] += 1
        if iteration[1] % 100 == 0
            println("Current loss is: $l, iteration is $(iteration[1])")
        end
        if haslogger
            log_value(logger, "outer_error/loss", l, step=iteration[1])
            if iteration[1] % 30 == 0
                u_predict = reshape([first(phi([x,y],p)) for x in xs for y in ys],(length(xs),length(ys)))
                diff_u = abs.(u_predict .- u_real)
                total_diff = sum(diff_u)
                log_value(logger, "outer_error/total_diff", total_diff, step=iteration[1])
                total_u = sum(abs.(u_real))
                total_diff_rel = total_diff / total_u
                log_value(logger, "outer_error/total_diff_rel", total_diff_rel, step=iteration[1])
                total_diff_sq = sum(diff_u .^ 2)
                log_value(logger, "outer_error/total_diff_sq", total_diff_sq, step=iteration[1])
            end
        end
        return false
    end
    res = GalacticOptim.solve(prob, ADAM(0.03); maxiters=maxiters, cb=cb)

    u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
    diff_u = abs.(u_predict .- u_real)
    total_diff = sum(diff_u)
    total_u = sum(abs.(u_real))
    total_diff_rel = total_diff / total_u

    #p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    #p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    #p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    #(plot=plot(p1,p2,p3), error=total_diff, total_diff_rel=total_diff_rel)
    (error=total_diff, total_diff_rel=total_diff_rel)
end



println("making sure that there are no logs without having imported NeuralPDELogging")
no_logger_loc = joinpath(@__DIR__, "testlogs", "no_logs")
test_2d_poisson_equation_adaptive_loss_no_logs_run_seediters(adaptive_loss, run) = test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, no_logger_loc, false; seed=seed, maxiters=maxiters_no_logs)
error_results_no_logs = map(test_2d_poisson_equation_adaptive_loss_no_logs_run_seediters, adaptive_losses, 1:length(adaptive_losses))

# make sure that that no logger is putting logs in there
@test length(readdir(no_logger_loc)) == 0

# this should recompile the logging stuff
using NeuralPDELogging

println("making sure that there are still no logs now that we have imported NeuralPDELogging")
no_logger_after_import_loc = joinpath(@__DIR__, "testlogs", "no_logs_after_import")
test_2d_poisson_equation_adaptive_loss_no_logs_after_import_run_seediters(adaptive_loss, run) = test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, no_logger_after_import_loc, false; seed=seed, maxiters=maxiters_no_logs)
error_results_no_logs = map(test_2d_poisson_equation_adaptive_loss_no_logs_after_import_run_seediters, adaptive_losses, 1:length(adaptive_losses))

# make sure that that no logger is putting logs in there
@test length(readdir(no_logger_after_import_loc)) == 0

logger_loc = joinpath(@__DIR__, "testlogs", "logs")
println("test logger logs going into folder:")


test_2d_poisson_equation_adaptive_loss_with_logs_run_seediters(adaptive_loss, run) = test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, logger_loc, true; seed=seed, maxiters=maxiters_logs)
error_results_with_logs = map(test_2d_poisson_equation_adaptive_loss_with_logs_run_seediters, adaptive_losses, 1:length(adaptive_losses))

@show error_results_with_logs[1][:total_diff_rel]
@show error_results_with_logs[2][:total_diff_rel]
@show error_results_with_logs[3][:total_diff_rel]
# accuracy tests, these work for this specific seed but might not for others
# note that this doesn't test that the adaptive losses are outperforming the nonadaptive loss, which is not guaranteed, and seed/arch/hyperparam/pde etc dependent
@test error_results_with_logs[1][:total_diff_rel] < 0.4
@test error_results_with_logs[2][:total_diff_rel] < 0.4
@test error_results_with_logs[3][:total_diff_rel] < 0.4
# make sure that the logger is actually putting logs in there, but not testing the contents of the logs
@test length(readdir(joinpath(loggerloc, "1"))) > 0
@test length(readdir(joinpath(loggerloc, "2"))) > 0
@test length(readdir(joinpath(loggerloc, "3"))) > 0


#plots_diffs[1][:plot]
#plots_diffs[2][:plot]
#plots_diffs[3][:plot]

