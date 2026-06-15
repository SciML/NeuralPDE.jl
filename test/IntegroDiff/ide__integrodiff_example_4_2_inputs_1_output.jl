module IntegroDiffTestSetup

    function callback(p, l)
        if p.iter == 1 || p.iter % 10 == 0
            println("Current loss is: $l after $(p.iter) iterations")
        end
        return false
    end

    export callback

end

using .IntegroDiffTestSetup

using NeuralPDE
using Test

@testset "IntegroDiff Example 4 -- 2 Inputs, 1 Output" begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x, y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Ix = Integral((x, y) in DomainSets.ProductDomain(UnitInterval(), ClosedInterval(0, x)))

    eq = Ix(u(x, y)) ~ 5 / 12
    bcs = [u(0.0, 0.0) ~ 0, Dy(u(x, y)) ~ 2 * y, u(x, 0) ~ x]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(2, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    phi = discretization.phi

    xs = 0.0:0.01:1.0
    ys = 0.0:0.01:1.0

    u_real = collect(x + y^2 for y in ys, x in xs)
    u_predict = collect(Array(phi([x, y], res.u))[1] for y in ys, x in xs)
    @test mean(abs2, u_real .- u_predict) < 0.02
end
