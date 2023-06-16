using Flux, NeuralPDE, Test
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Integrals, IntegralsCubature
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
import Lux

using Random
Random.seed!(100)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

## Example 1, 1D ode
function test_ode(strategy_)
    println("Example 1, 1D ode: strategy: $(nameof(typeof(strategy_)))")
    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~ θ^3 + 2 * θ + (θ^2) * ((1 + 3 * (θ^2)) / (1 + θ + (θ^3))) -
                    u(θ) * (θ + ((1 + 3 * (θ^2)) / (1 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = Lux.Chain(Lux.Dense(1, 12, Flux.σ), Lux.Dense(12, 1))

    discretization = NeuralPDE.PhysicsInformedNN(chain,
                                                 strategy_)

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u(θ)])
    prob = NeuralPDE.discretize(pde_system, discretization)

    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.1); maxiters = 1000)
    prob = remake(prob, u0 = res.minimizer)
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 500)
    prob = remake(prob, u0 = res.minimizer)
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); maxiters = 500)
    phi = discretization.phi

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(phi(t, res.minimizer)) for t in ts]

    @test u_predict≈u_real atol=0.1
    # using Plots
    # t_plot = collect(ts)
    # plot(t_plot ,u_real)
    # plot!(t_plot ,u_predict)
end

#TODO There is little meaning in these tests without checking the correctness of the prediction.
#TODO I propose to simply remove them.
function test_heterogeneous_equation(strategy_)
    println("Simple Heterogeneous input PDE, strategy: $(nameof(typeof(strategy_)))")
    @parameters x y
    @variables p(..) q(..) r(..) s(..)
    Dx = Differential(x)
    Dy = Differential(y)

    # 2D PDE
    eq = p(x) + q(y) + Dx(r(x, y)) + Dy(s(y, x)) ~ 0
    # eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

    # Initial and boundary conditions
    bcs = [p(1) ~ 0.0f0, q(-1) ~ 0.0f0,
        r(x, -1) ~ 0.0f0, r(1, y) ~ 0.0f0,
        s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
    # bcs = [s(y, 1) ~ 0.0f0]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    # chain_ = Lux.Chain(Lux.Dense(2,12,Flux.σ),Lux.Dense(12,12,Flux.σ),Lux.Dense(12,1))
    numhid = 3
    luxchain = [[Lux.Chain(Lux.Dense(1, numhid, Flux.σ),
                           Lux.Dense(numhid, numhid, Flux.σ), Lux.Dense(numhid, 1))
                 for i in 1:2]
                [Lux.Chain(Lux.Dense(2, numhid, Flux.σ),
                           Lux.Dense(numhid, numhid, Flux.σ), Lux.Dense(numhid, 1))
                 for i in 1:2]]
    discretization = NeuralPDE.PhysicsInformedNN(luxchain,
                                                 strategy_)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [p(x), q(y), r(x, y), s(y, x)])
    prob = SciMLBase.discretize(pde_system, discretization)
    res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 100)
end

## Heterogeneous system
function test_heterogeneous_system(strategy_)
    println("Heterogeneus input PDE with derivatives, strategy: $(nameof(typeof(strategy_)))")
    @parameters x y
    @variables p(..) q(..)
    Dx = Differential(x)
    Dy = Differential(y)

    # 2D PDE
    #TODO Dx(q(y)) = 0
    #TODO so p(x) = 0, q = const is has only trivial solution
    eq = p(x) + Dx(q(y)) ~ 0

    # Initial and boundary conditions
    bcs = [p(1) ~ 0.0f0, q(-1) ~ 0.0f0]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(-1.0, 0.0)]

    # chain_ = Lux.Chain(Lux.Dense(2,12,Flux.σ),Lux.Dense(12,12,Flux.σ),Lux.Dense(12,1))
    numhid = 3
    luxchain = [[Lux.Chain(Lux.Dense(1, numhid, Flux.σ),
                           Lux.Dense(numhid, numhid, Flux.σ), Lux.Dense(numhid, 1))
                 for i in 1:2]
                [Lux.Chain(Lux.Dense(2, numhid, Flux.σ),
                           Lux.Dense(numhid, numhid, Flux.σ), Lux.Dense(numhid, 1))
                 for i in 1:2]]
    discretization = NeuralPDE.PhysicsInformedNN(luxchain,
                                                 strategy_)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [p(x), q(y)])
    prob = SciMLBase.discretize(pde_system, discretization)
    res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 100)
end

grid_strategy = NeuralPDE.GridTraining(0.1)
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(),
                                                   reltol = 1e3, abstol = 1e-3,
                                                   maxiters = 50, batch = 100)
stochastic_strategy = NeuralPDE.StochasticTraining(100; bcs_points = 50)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100;
                                                     sampling_alg = LatinHypercubeSample(),
                                                     resampling = false,
                                                     minibatch = 100)
quasirandom_strategy_resampling = NeuralPDE.QuasiRandomTraining(100;
                                                                bcs_points = 50,
                                                                sampling_alg = LatticeRuleSample(),
                                                                resampling = true,
                                                                minibatch = 0)

strategies = [
    grid_strategy,
    stochastic_strategy,
    quasirandom_strategy,
    quasirandom_strategy_resampling,
    quadrature_strategy,
]
@testset "Test ODE/Heterogeneous" begin

map(strategies) do strategy_
    test_ode(strategy_)
end end

## Heterogeneous system
@testset "Example 1: Heterogeneous system" begin
    @parameters x, y, z
    @variables u(..), v(..), h(..), p(..)
    Dz = Differential(z)
    eqs = [
        u(x, y, z) ~ x + y + z,
        v(y, x) ~ x^2 + y^2,
        h(z) ~ cos(z),
        p(x, z) ~ exp(x) * exp(z),
        u(x, y, z) + v(y, x) * Dz(h(z)) - p(x, z) ~ x + y + z - (x^2 + y^2) * sin(z) -
                                                    exp(x) * exp(z),
    ]

    bcs = [u(0, 0, 0) ~ 0.0]

    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
        z ∈ Interval(0.0, 1.0)]

    chain = [
        Lux.Chain(Lux.Dense(3, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
                  Lux.Dense(12, 1)),
        Lux.Chain(Lux.Dense(2, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
                  Lux.Dense(12, 1)),
        Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
                  Lux.Dense(12, 1)),
        Lux.Chain(Lux.Dense(2, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
                  Lux.Dense(12, 1))]

    grid_strategy = NeuralPDE.GridTraining(0.1)
    quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(),
                                                       reltol = 1e-3, abstol = 1e-3,
                                                       maxiters = 50, batch = 100)

    discretization = NeuralPDE.PhysicsInformedNN(chain, grid_strategy)

    @named pde_system = PDESystem(eqs, bcs, domains, [x, y, z],
                                  [u(x, y, z), v(y, x), h(z), p(x, z)])

    prob = NeuralPDE.discretize(pde_system, discretization)

    callback = function (p, l)
        println("Current loss is: $l")
        return false
    end

    res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 2000)

    phi = discretization.phi

    analytic_sol_func_ = [
        (x, y, z) -> x + y + z,
        (x, y) -> x^2 + y^2,
        (z) -> cos(z),
        (x, z) -> exp(x) * exp(z),
    ]

    xs, ys, zs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

    u_real = [analytic_sol_func_[1](x, y, z) for x in xs for y in ys for z in zs]
    v_real = [analytic_sol_func_[2](y, x) for y in ys for x in xs]
    h_real = [analytic_sol_func_[3](z) for z in zs]
    p_real = [analytic_sol_func_[4](x, z) for x in xs for z in zs]

    real_ = [u_real, v_real, h_real, p_real]

    u_predict = [phi[1]([x, y, z], res.u.depvar.u)[1] for x in xs for y in ys
                 for z in zs]
    v_predict = [phi[2]([y, x], res.u.depvar.v)[1] for y in ys for x in xs]
    h_predict = [phi[3]([z], res.u.depvar.h)[1] for z in zs]
    p_predict = [phi[4]([x, z], res.u.depvar.p)[1] for x in xs for z in zs]
    predict = [u_predict, v_predict, h_predict, p_predict]

    for i in 1:4
        @test predict[i]≈real_[i] rtol=10^-2
    end

    # x_plot = collect(xs)
    # y_plot = collect(ys)
    # i=1
    # z=0
    # u_real = collect(analytic_sol_func_[1](x,y,z) for y in ys, x in xs);
    # u_predict = collect(phi[1]([x,y,z],minimizers[1])[1]  for y in ys, x in xs);
    # plot(x_plot,y_plot,u_real)
    # plot!(x_plot,y_plot,u_predict)
end

## Example 2, 2D Poisson equation
function test_2d_poisson_equation(chain_, strategy_)
    println("Example 2, 2D Poisson equation, chain: $(nameof(typeof(chain_))), strategy: $(nameof(typeof(strategy_)))")
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

    discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                 strategy_)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = NeuralPDE.discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.1); maxiters = 500,
                             cb = callback)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

    u_predict = reshape([first(phi([x, y], res.minimizer)) for x in xs for y in ys],
                        (length(xs), length(ys)))
    u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                     (length(xs), length(ys)))
    diff_u = abs.(u_predict .- u_real)

    @test u_predict≈u_real atol=2.0

    # p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    # p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    # p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    # plot(p1,p2,p3)
end

@testset "Example 2, 2D Poisson equation" begin
    grid_strategy = GridTraining(0.1)

    chain = Lux.Chain(Lux.Dense(2, 12, Flux.σ), Lux.Dense(12, 12, Flux.σ), Lux.Dense(12, 1))
    fluxchain = Chain(Dense(2, 12, Flux.σ), Dense(12, 12, Flux.σ), Dense(12, 1)) |> f64
    chains = [fluxchain, chain]
    for chain in chains
        test_2d_poisson_equation(chain, grid_strategy)
    end

    for strategy_ in strategies
        chain_ = Lux.Chain(Lux.Dense(2, 12, Flux.σ), Lux.Dense(12, 12, Flux.σ),
                           Lux.Dense(12, 1))
        test_2d_poisson_equation(chain_, strategy_)
    end

    algs = [CubatureJLp()] #CubatureJLh(),
    for alg in algs
        chain_ = Lux.Chain(Lux.Dense(2, 12, Flux.σ), Lux.Dense(12, 12, Flux.σ),
                           Lux.Dense(12, 1))
        strategy_ = NeuralPDE.QuadratureTraining(quadrature_alg = alg, reltol = 1e-4,
                                                 abstol = 1e-3, maxiters = 30, batch = 10)
        test_2d_poisson_equation(chain_, strategy_)
    end
end

## Example 3, 3rd-order
@testset "Example 3, 3rd-order ode" begin
    @parameters x
    @variables u(..), Dxu(..), Dxxu(..), O1(..), O2(..)
    Dxxx = Differential(x)^3
    Dx = Differential(x)

    # ODE
    eq = Dx(Dxxu(x)) ~ cos(pi * x)

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 0.0,
        u(1.0) ~ cos(pi),
        Dxu(1.0) ~ 1.0]
    ep = (cbrt(eps(eltype(Float64))))^2 / 6

    der = [Dxu(x) ~ Dx(u(x)) + ep * O1(x),
        Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x)]

    eqs = [eq; der]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = [[Lux.Chain(Lux.Dense(1, 12, Lux.tanh), Lux.Dense(12, 12, Lux.tanh),
                        Lux.Dense(12, 1)) for _ in 1:3]
             [Lux.Chain(Lux.Dense(1, 4, Lux.tanh), Lux.Dense(4, 1)) for _ in 1:2]]
    quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100; #points
                                                         sampling_alg = LatinHypercubeSample())

    discretization = NeuralPDE.PhysicsInformedNN(chain, quasirandom_strategy)

    @named pde_system = PDESystem(eqs, bcs, domains, [x],
                                  [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)])

    prob = NeuralPDE.discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

    cb_ = function (p, l)
        println("loss: ", l)
        println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
        println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
        return false
    end

    res = solve(prob, OptimizationOptimJL.BFGS(); maxiters = 1000)
    phi = discretization.phi[1]

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sin(π * x)) / (π^3)

    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u.depvar.u)) for x in xs]

    @test u_predict≈u_real atol=10^-4

    # x_plot = collect(xs)
    # plot(x_plot ,u_real)
    # plot!(x_plot ,u_predict)
end
## Example 4, system of pde
@testset "Example 4, system of pde" begin
    @parameters x, y
    @variables u1(..), u2(..)
    Dx = Differential(x)
    Dy = Differential(y)

    # System of pde
    eqs = [Dx(u1(x, y)) + 4 * Dy(u2(x, y)) ~ 0,
        Dx(u2(x, y)) + 9 * Dy(u1(x, y)) ~ 0]
    # 3*u1(x,0) ~ 2*u2(x,0)]

    # Initial and boundary conditions
    bcs = [u1(x, 0) ~ 2 * x, u2(x, 0) ~ 3 * x]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # Neural network
    chain1 = Lux.Chain(Lux.Dense(2, 15, Lux.tanh), Lux.Dense(15, 1))
    chain2 = Lux.Chain(Lux.Dense(2, 15, Lux.tanh), Lux.Dense(15, 1))

    quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(),
                                                       reltol = 1e-3, abstol = 1e-3,
                                                       maxiters = 50, batch = 100)
    chain = [chain1, chain2]

    discretization = NeuralPDE.PhysicsInformedNN(chain, quadrature_strategy)

    @named pde_system = PDESystem(eqs, bcs, domains, [x, y], [u1(x, y), u2(x, y)])

    prob = NeuralPDE.discretize(pde_system, discretization)

    res = solve(prob, OptimizationOptimJL.BFGS(); maxiters = 1000)
    phi = discretization.phi

    analytic_sol_func(x, y) = [1 / 3 * (6x - y), 1 / 2 * (6x - y)]
    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    u_real = [[analytic_sol_func(x, y)[i] for x in xs for y in ys] for i in 1:2]
    depvars = [:u1, :u2]

    u_predict = [[phi[i]([x, y], res.u.depvar[depvars[i]])[1] for x in xs for y in ys]
                 for i in 1:2]

    @test u_predict[1]≈u_real[1] atol=0.1
    @test u_predict[2]≈u_real[2] atol=0.1

    # p1 =plot(xs, ys, u_predict, st=:surface);
    # p2 = plot(xs, ys, u_real, st=:surface);
    # plot(p1,p2)
end

# ## Example 5, 2d wave equation, neumann boundary condition
# @testset "Example 5, 2d wave equation, neumann boundary condition" begin
#     #here we use low level api for build solution
#     @parameters x, t
#     @variables u(..)
#     Dxx = Differential(x)^2
#     Dtt = Differential(t)^2
#     Dt = Differential(t)

#     #2D PDE
#     C = 1
#     eq = Dtt(u(x, t)) ~ C^2 * Dxx(u(x, t))

#     # Initial and boundary conditions
#     bcs = [u(0, t) ~ 0.0,# for all t > 0
#         u(1, t) ~ 0.0,# for all t > 0
#         u(x, 0) ~ x * (1.0 - x), #for all 0 < x < 1
#         Dt(u(x, 0)) ~ 0.0] #for all  0 < x < 1]

#     # Space and time domains
#     domains = [x ∈ Interval(0.0, 1.0),
#         t ∈ Interval(0.0, 1.0)]
#     @named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

#     # Neural network
#     chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ), Lux.Dense(16, 16, Lux.σ), Lux.Dense(16, 1))
#     phi = NeuralPDE.Phi(chain)
#     derivative = NeuralPDE.numeric_derivative

#     quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(),
#                                                        reltol = 1e-3, abstol = 1e-3,
#                                                        maxiters = 50, batch = 100)

#     discretization = NeuralPDE.PhysicsInformedNN(chain, quadrature_strategy)
#     prob = NeuralPDE.discretize(pde_system, discretization)

#     cb_ = function (p, l)
#         println("loss: ", l)
#         println("losses: ", map(l -> l(p), loss_functions))
#         return false
#     end

#     res = solve(prob, OptimizationOptimJL.BFGS(); maxiters = 500, f_abstol = 10^-6)

#     dx = 0.1
#     xs, ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
#     function analytic_sol_func(x, t)
#         sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50000])
#     end

#     u_predict = reshape([first(phi([x, t], res.u)) for x in xs for t in ts],
#                         (length(xs), length(ts)))
#     u_real = reshape([analytic_sol_func(x, t) for x in xs for t in ts],
#                      (length(xs), length(ts)))

#     @test u_predict≈u_real atol=0.1

#     # diff_u = abs.(u_predict .- u_real)
#     # p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
#     # p2 =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
#     # p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
#     # plot(p1,p2,p3)
# end
# ## Example 6, pde with mixed derivative
# @testset "Example 6, pde with mixed derivative" begin
#     @parameters x y
#     @variables u(..)
#     Dxx = Differential(x)^2
#     Dyy = Differential(y)^2
#     Dx = Differential(x)
#     Dy = Differential(y)

#     eq = Dxx(u(x, y)) + Dx(Dy(u(x, y))) - 2 * Dyy(u(x, y)) ~ -1.0

#     # Initial and boundary conditions
#     bcs = [u(x, 0) ~ x,
#         Dy(u(x, 0)) ~ x,
#         u(x, 0) ~ Dy(u(x, 0))]

#     # Space and time domains
#     domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

#     quadrature_strategy = NeuralPDE.QuadratureTraining()
#     # Neural network
#     inner = 20
#     chain = Lux.Chain(Lux.Dense(2, inner, Lux.tanh), Lux.Dense(inner, inner, Lux.tanh),
#                       Lux.Dense(inner, 1))

#     discretization = NeuralPDE.PhysicsInformedNN(chain, quadrature_strategy)
#     @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

#     prob = NeuralPDE.discretize(pde_system, discretization)

#     res = solve(prob, OptimizationOptimJL.BFGS(); maxiters = 1500)
#     @show res.original

#     phi = discretization.phi

#     analytic_sol_func(x, y) = x + x * y + y^2 / 2
#     xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

#     u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
#                         (length(xs), length(ys)))
#     u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
#                      (length(xs), length(ys)))
#     diff_u = abs.(u_predict .- u_real)

#     @test u_predict≈u_real rtol=0.1

#     # p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
#     # p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
#     # p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
#     # plot(p1,p2,p3)
# end
