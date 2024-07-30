using NeuralPDE, DomainSets, Lux
using LinearAlgebra
using Random, ComponentArrays
using OptimizationOptimisers

@testset "Continuous" begin
    #Base 

    tension = LinRange(0.1, 10, 100)
    callback = function (p, l)
        println("Current loss is: $l")
        return false
    end

    #constants
    L = 1
    ei = 1.0
    T = 0.0
    Q = [0, -0.5, 0]

    @parameters s
    @variables theta(..)
    Ds = Differential(s)
    Dss = Differential(s)^2
    I_0_l = Integral(s in DomainSets.ClosedInterval(0, L))
    I_0_s = Integral(s in DomainSets.ClosedInterval(0, s))

    P = [I_0_l(cos(theta(s))), I_0_l(sin(theta(s))), 0]
    A = [I_0_s(cos(theta(s))), I_0_s(sin(theta(s))), 0]
    F = ((Q - P) / (norm(Q - P))) * T
    t = [cos(theta(s)), sin(theta(s)), 0]

    eq = ei * Dss(theta(s)) + dot(cross(t, F), [0, 0, 1]) ~ 0
    bcs = [theta(0.0) ~ 0.0, Ds(theta(L)) ~ 0.0]
    domains = [s ∈ Interval(0.0, 1.0)]

    strategy_ = QuadratureTraining()
    af = Lux.relu
    chain1 = Chain(Dense(1, 10, af), Dense(10, 1)) |> f64
    loss = []

    callback = function (p, l)
        println("loss: $l")
        append!(loss, l)
        return false
    end

    init_params = Lux.setup(Random.default_rng(), chain1)[1] |> ComponentArray .|> Float64
    discretization = NeuralPDE.PhysicsInformedNN(
        chain1, strategy_; init_params = init_params)

    @named pde_system = PDESystem(eq, bcs, domains, [s], [theta(s)])
    prob = NeuralPDE.discretize(pde_system, discretization)

    @time res = Optimization.solve(
        prob, OptimizationOptimisers.Adam(5e-3); callback = callback, maxiters = 200) # 1.69s

    T = 0.1
    F = ((Q - P) / (norm(Q - P))) .* T
    eq = ei * Dss(theta(s)) + dot(cross(t, F), [0, 0, 1]) ~ 0

    discretization = NeuralPDE.PhysicsInformedNN(chain1, strategy_; init_params = res.u)
    @named pde_system = PDESystem(eq, bcs, domains, [s], [theta(s)])
    prob = NeuralPDE.discretize(pde_system, discretization)
    @time res2 = Optimization.solve(
        prob, OptimizationOptimisers.Adam(5e-3); callback = callback, maxiters = 200) # 63.9s

    discretization = NeuralPDE.PhysicsInformedNN(chain1, strategy_)
    @named pde_system = PDESystem(eq, bcs, domains, [s], [theta(s)])
    prob = NeuralPDE.discretize(pde_system, discretization)
    @time res2 = Optimization.solve(
        prob, OptimizationOptimisers.Adam(5e-3); callback = callback, maxiters = 200) # 114.58s
end

#########################

# Issue 733

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, Random
import ModelingToolkit: Interval

@parameters x, y, t
@variables u(..)

L = 1
tmax = 1
domains = [x ∈ Interval(-L, +L),
    y ∈ Interval(-L, +L),
    t ∈ Interval(0, tmax)]

Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dtt = Differential(t)^2

eq = Dtt(u(x, y, t)) ~ Dxx(u(x, y, t)) + Dyy(u(x, y, t))

bcs = [u(x, y, 0) ~ cos(x) * cosh(y), u(-1, y, t) ~ u(L, y, t), u(x, -L, t) ~ u(x, L, t),
    Dx(u(-L, y, t)) ~ Dx(u(L, y, t)), Dy(u(x, -L, t)) ~ Dy(u(x, L, t))]

in = length(domains)
n = 9
chain = Lux.Chain(
    Dense(in, n, Lux.asinh), Dense(n, n, Lux.asinh), Dense(n, n, Lux.asinh), Dense(n, 1))

discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pdesystem = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t)])

prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)
callback = function (p, l)
    println("loss: ", l)
    println(" pde: ", map(l_ -> l_(p.u), sym_prob.loss_functions.pde_loss_functions))
    println(" bcs: ", map(l_ -> l_(p.u), sym_prob.loss_functions.bc_loss_functions))
    println()
    return false
end

inter = 1
res = Optimization.solve(prob, Adam(0.1); callback = callback, maxiters = inter)
