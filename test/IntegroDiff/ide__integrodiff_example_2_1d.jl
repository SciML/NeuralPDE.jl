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

@testset "IntegroDiff Example 2 -- 1D" begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x
    @variables u(..)
    Ix = Integral(x in DomainSets.ClosedInterval(0, x))

    eq = Ix(u(x) * cos(x)) ~ (x^3) / 3
    bcs = [u(0.0) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]

    chain = Chain(Dense(1, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi

    u_real = [x^2 / cos(x) for x in xs]
    u_predict = [first(phi([x], res.u)) for x in xs]
    @test mean(abs2, u_real .- u_predict) < 0.02
end
