@testsetup module AdaptiveLossTestSetup
using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, NeuralPDE, Test,
      TensorBoardLogger
import DomainSets: Interval, infimum, supremum

function solve_with_adaptive_loss(
        adaptive_loss; haslogger = false, outdir = mktempdir(), run = 1)
    logdir = joinpath(outdir, string(run))
    logger = haslogger ? TBLogger(logdir) : nothing

    Random.seed!(60)
    hid = 40
    chain = Chain(Dense(2, hid, tanh), Dense(hid, hid, tanh), Dense(hid, 1))
    strategy = StochasticTraining(256)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

    # Initial and boundary conditions
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ -sinpi(1) * sinpi(y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sinpi(x) * sinpi(1)
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    discretization = PhysicsInformedNN(chain, strategy; adaptive_loss, logger)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)
    u_real = [analytic_sol_func(x, y) for x in xs for y in ys]

    callback = function (p, l)
        if p.iter % 250 == 0
            @info "[$(nameof(typeof(adaptive_loss)))] Current loss is: $l, iteration is $(p.iter)"
        end
        if haslogger
            log_value(logger, "outer_error/loss", l, step = p.iter)
            if p.iter % 30 == 0
                u_predict = [first(phi([x, y], p.u)) for x in xs for y in ys]
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

    res = solve(prob, Adam(0.03); maxiters = 2000, callback)
    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]

    total_diff = sum(abs, u_predict .- u_real)
    total_u = sum(abs, u_real)
    total_diff_rel = total_diff / total_u

    return total_diff_rel
end

export solve_with_adaptive_loss

end

@testitem "2D Poisson: NonAdaptiveLoss" tags=[:adaptiveloss] setup=[AdaptiveLossTestSetup] begin
    loss=NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)

    tmpdir=mktempdir()

    total_diff_rel=solve_with_adaptive_loss(
        loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
    @test length(readdir(tmpdir)) == 0

    total_diff_rel=solve_with_adaptive_loss(
        loss; haslogger = true, outdir = tmpdir, run = 2)
    @test total_diff_rel < 0.4
    @test length(readdir(tmpdir)) == 1
end

@testitem "2D Poisson: GradientScaleAdaptiveLoss" tags=[:adaptiveloss] setup=[AdaptiveLossTestSetup] begin
    loss=GradientScaleAdaptiveLoss(100, pde_loss_weights = 1e3, bc_loss_weights = 1)

    tmpdir=mktempdir()

    total_diff_rel=solve_with_adaptive_loss(
        loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
    @test length(readdir(tmpdir)) == 0

    total_diff_rel=solve_with_adaptive_loss(
        loss; haslogger = true, outdir = tmpdir, run = 2)
    @test total_diff_rel < 0.4
    @test length(readdir(tmpdir)) == 1
end

@testitem "2D Poisson: MiniMaxAdaptiveLoss" tags=[:adaptiveloss] setup=[AdaptiveLossTestSetup] begin
    loss=MiniMaxAdaptiveLoss(100; pde_loss_weights = 1, bc_loss_weights = 1)

    tmpdir=mktempdir()

    total_diff_rel=solve_with_adaptive_loss(
        loss; haslogger = false, outdir = tmpdir, run = 1)
    @test total_diff_rel < 0.4
    @test length(readdir(tmpdir)) == 0

    total_diff_rel=solve_with_adaptive_loss(
        loss; haslogger = true, outdir = tmpdir, run = 2)
    @test total_diff_rel < 0.4
    @test length(readdir(tmpdir)) == 1
end
