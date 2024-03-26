@info "adaptive_loss_logging_tests"
using Test, NeuralPDE
using Optimization, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum
using Random, Lux
@info "Starting Soon!"

nonadaptive_loss = NeuralPDE.NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)
gradnormadaptive_loss = NeuralPDE.GradientScaleAdaptiveLoss(100, pde_loss_weights = 1e3,
                                                            bc_loss_weights = 1)
adaptive_loss = NeuralPDE.MiniMaxAdaptiveLoss(100; pde_loss_weights = 1,
                                              bc_loss_weights = 1)
adaptive_losses = [nonadaptive_loss, gradnormadaptive_loss, adaptive_loss]
maxiters = 800
seed = 60

## 2D Poisson equation
function test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, outdir, haslogger;
                                                seed = 60, maxiters = 800)
    logdir = joinpath(outdir, string(run))
    if haslogger
        logger = TBLogger(logdir)
    else
        logger = nothing
    end
    Random.seed!(seed)
    hid = 40
    chain_ = Lux.Chain(Dense(2, hid, Lux.σ), Dense(hid, hid, Lux.σ),
                       Dense(hid, 1))
    strategy_ = NeuralPDE.StochasticTraining(256)
    @info "adaptive reweighting test logdir: $(logdir), maxiters: $(maxiters), 2D Poisson equation, adaptive_loss: $(nameof(typeof(adaptive_loss))) "
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
                                                 logger = logger,
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
        if haslogger
            log_value(logger, "outer_error/loss", l, step = iteration[1])
            if iteration[1] % 30 == 0
                u_predict = reshape([first(phi([x, y], p.u)) for x in xs for y in ys],
                                    (length(xs), length(ys)))
                diff_u = abs.(u_predict .- u_real)
                total_diff = sum(diff_u)
                log_value(logger, "outer_error/total_diff", total_diff, step = iteration[1])
                total_u = sum(abs.(u_real))
                total_diff_rel = total_diff / total_u
                log_value(logger, "outer_error/total_diff_rel", total_diff_rel,
                          step = iteration[1])
                total_diff_sq = sum(diff_u .^ 2)
                log_value(logger, "outer_error/total_diff_sq", total_diff_sq,
                          step = iteration[1])
            end
        end
        return false
    end
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.03); maxiters = maxiters,
                             callback = callback)

    u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
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

possible_logger_dir = mktempdir()
if ENV["LOG_SETTING"] == "NoImport"
    haslogger = false
    expected_log_folders = 0
elseif ENV["LOG_SETTING"] == "ImportNoUse"
    using NeuralPDELogging
    haslogger = false
    expected_log_folders = 0
elseif ENV["LOG_SETTING"] == "ImportUse"
    using NeuralPDELogging
    using TensorBoardLogger
    haslogger = true
    expected_log_folders = 3
end

@info "has logger: $(haslogger), expected log folders: $(expected_log_folders)"

function test_2d_poisson_equation_adaptive_loss_run_seediters(adaptive_loss, run)
    test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, possible_logger_dir,
                                           haslogger; seed = seed, maxiters = maxiters)
end
error_results = map(test_2d_poisson_equation_adaptive_loss_run_seediters, adaptive_losses,
                    1:length(adaptive_losses))

@test length(readdir(possible_logger_dir)) == expected_log_folders
if expected_log_folders > 0
    @info "dirs at $(possible_logger_dir): $(string(readdir(possible_logger_dir)))"
    for logdir in readdir(possible_logger_dir)
        @test length(readdir(joinpath(possible_logger_dir, logdir))) > 0
    end
end
