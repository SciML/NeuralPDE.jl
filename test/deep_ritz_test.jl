using NeuralPDE, Test

using ModelingToolkit, Optimization, OptimizationOptimisers, Distributions, MethodOfLines,
      OrdinaryDiffEq
import ModelingToolkit: Interval, infimum, supremum
using Lux #: tanh, identity

@testset "Poisson's equation" begin
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
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    strategy = QuasiRandomTraining(256, minibatch = 32)
    hid = 40
    chain_ = Lux.Chain(Lux.Dense(2, hid, Lux.σ), Lux.Dense(hid, hid, Lux.σ),
        Lux.Dense(hid, 1))
    discretization = DeepRitz(chain_, strategy);

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)

    global iter = 0
    callback = function (p, l)
        global iter += 1
        if iter % 50 == 0
            println("$iter => $l")
        end
        return false
    end

    res = Optimization.solve(prob, Adam(0.01); callback = callback, maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = Optimization.solve(prob, Adam(0.001); callback = callback, maxiters = 200)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

    u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
        (length(xs), length(ys)))
    u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
        (length(xs), length(ys)))
    @test u_predict≈u_real atol=0.1
end