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

@testset "IntegroDiff Example 7: Infinity" begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(x, Inf))

    eq = I(u(x)) ~ 1 / x
    domains = [x ∈ Interval(1.0, 2.0)]
    bcs = [u(1) ~ 1]
    chain = Chain(Dense(1, 12, tanh), Dense(12, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 300)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    u_predict = [first(phi([x], res.u)) for x in xs]
    u_real = [1 / x^2 for x in xs]
    @test u_real ≈ u_predict rtol = 0.02
end
