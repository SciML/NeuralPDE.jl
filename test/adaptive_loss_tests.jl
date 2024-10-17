using Optimization, OptimizationOptimisers, Test, NeuralPDE, Random, DomainSets, Lux
import ModelingToolkit: Interval, infimum, supremum

nonadaptive_loss = NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 1)
gradnormadaptive_loss = GradientScaleAdaptiveLoss(100, pde_loss_weights = 1e3,
    bc_loss_weights = 1)
adaptive_loss = MiniMaxAdaptiveLoss(100; pde_loss_weights = 1, bc_loss_weights = 1)
adaptive_losses = [nonadaptive_loss, gradnormadaptive_loss, adaptive_loss]
maxiters = 4000
seed = 60

## 2D Poisson equation
function test_2d_poisson_equation_adaptive_loss(adaptive_loss; seed = 60, maxiters = 4000)
    Random.seed!(seed)
    hid = 32
    chain_ = Chain(Dense(2, hid, tanh), Dense(hid, hid, tanh), Dense(hid, 1))

    strategy_ = StochasticTraining(256)

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
    discretization = PhysicsInformedNN(chain_, strategy_; adaptive_loss, logger = nothing,
        iteration)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    phi = discretization.phi
    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)
    u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
        (length(xs), length(ys)))

    callback = function (p, l)
        iteration[] += 1
        if iteration[] % 100 == 0
            @info "Current loss is: $l, iteration is $(iteration[])"
        end
        return false
    end
    res = solve(prob, OptimizationOptimisers.Adam(0.03); maxiters, callback)
    u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
        (length(xs), length(ys)))
    total_diff = sum(abs, u_predict .- u_real)
    total_u = sum(abs, u_real)
    total_diff_rel = total_diff / total_u
    return (; error = total_diff, total_diff_rel)
end

@testset "$(nameof(typeof(adaptive_loss)))" for adaptive_loss in adaptive_losses
    error_results_no_logs = test_2d_poisson_equation_adaptive_loss(
        adaptive_loss; seed, maxiters)

    @test error_results_no_logs[:total_diff_rel] < 0.4
end
