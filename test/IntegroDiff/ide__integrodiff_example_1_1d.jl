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

@testset "IntegroDiff Example 1 -- 1D" begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters t
    @variables i(..)
    Di = Differential(t)
    Ii = Integral(t in DomainSets.ClosedInterval(0, t))
    eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(t)) ~ 1
    bcs = [i(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chain = Chain(Dense(1, 15, σ), Dense(15, 1))
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, BFGS(); callback, maxiters = 100)
    ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    analytic_sol_func(t) = 1 / 2 * (exp(-t)) * (sin(2 * t))

    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(phi([t], res.u)) for t in ts]
    @test mean(abs2, u_real .- u_predict) < 0.02
end
