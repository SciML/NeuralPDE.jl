using Test, NeuralPDE, Optimization, OptimizationOptimisers, Random, Lux
import ModelingToolkit: Interval, infimum, supremum

nonadaptive_loss = NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)
gradnormadaptive_loss = GradientScaleAdaptiveLoss(100, pde_loss_weights = 1e3,
    bc_loss_weights = 1)
adaptive_loss = MiniMaxAdaptiveLoss(100; pde_loss_weights = 1, bc_loss_weights = 1)
adaptive_losses = [nonadaptive_loss, gradnormadaptive_loss, adaptive_loss]

possible_logger_dir = mktempdir()
if ENV["LOG_SETTING"] == "NoImport"
    haslogger = false
    expected_log_folders = 0
elseif ENV["LOG_SETTING"] == "ImportNoUse"
    using TensorBoardLogger
    haslogger = false
    expected_log_folders = 0
elseif ENV["LOG_SETTING"] == "ImportUse"
    using TensorBoardLogger
    haslogger = true
    expected_log_folders = 3
end

@info "has logger: $(haslogger), expected log folders: $(expected_log_folders)"

function test_2d_poisson_equation_adaptive_loss(adaptive_loss, run, outdir, haslogger;
        seed = 60, maxiters = 800)
    logdir = joinpath(outdir, string(run))
    logger = haslogger ? TBLogger(logdir) : nothing

    Random.seed!(seed)
    hid = 40
    chain_ = Chain(Dense(2, hid, σ), Dense(hid, hid, σ), Dense(hid, 1))
    strategy_ = StochasticTraining(256)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

    # Initial and boundary conditions
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sinpi(1) * sinpi(y),
        u(x, 0) ~ 0.0, u(x, 1) ~ -sinpi(x) * sinpi(1)]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    discretization = PhysicsInformedNN(chain_, strategy_; adaptive_loss, logger)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = NeuralPDE.discretize(pde_system, discretization)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    sz = (length(xs), length(ys))
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)
    u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys], sz)

    callback = function (p, l)
        if p.iter % 100 == 0
            @info "Current loss is: $l, iteration is $(p.iter)"
        end
        if haslogger
            log_value(logger, "outer_error/loss", l, step = p.iter)
            if p.iter % 30 == 0
                u_predict = reshape([first(phi([x, y], p.u)) for x in xs for y in ys],
                    (length(xs), length(ys)))
                total_diff = sum(abs, u_predict .- u_real)
                log_value(logger, "outer_error/total_diff", total_diff, step = p.iter)
                log_value(logger, "outer_error/total_diff_rel",
                    total_diff / sum(abs2, u_real), step = p.iter)
                log_value(logger, "outer_error/total_diff_sq",
                    sum(abs2, u_predict .- u_real), step = p.iter)
            end
        end
        return false
    end
    res = solve(prob, OptimizationOptimisers.Adam(0.03); maxiters, callback)

    u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys], sz)
    diff_u = abs.(u_predict .- u_real)
    total_diff = sum(diff_u)
    total_u = sum(abs.(u_real))
    total_diff_rel = total_diff / total_u

    return (error = total_diff, total_diff_rel = total_diff_rel)
end

@testset "$(nameof(typeof(adaptive_loss)))" for (i, adaptive_loss) in enumerate(adaptive_losses)
    test_2d_poisson_equation_adaptive_loss(adaptive_loss, i, possible_logger_dir,
        haslogger; seed = 60, maxiters = 800)
end

@test length(readdir(possible_logger_dir)) == expected_log_folders
if expected_log_folders > 0
    @info "dirs at $(possible_logger_dir): $(string(readdir(possible_logger_dir)))"
    for logdir in readdir(possible_logger_dir)
        @test length(readdir(joinpath(possible_logger_dir, logdir))) > 0
    end
end
