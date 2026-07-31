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

@testset "IntegroDiff Example 5 -- 1 Input, 2 Outputs" begin
    using Optimization, Optimisers, DomainSets, Lux, Random, Statistics
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x
    @variables u(..) w(..)
    Dx = Differential(x)
    Ix = Integral(x in DomainSets.ClosedInterval(1, x))

    eqs = [Ix(u(x) * w(x)) ~ log(abs(x)), Dx(w(x)) ~ -2 / (x^3), u(x) ~ x]
    bcs = [u(1.0) ~ 1.0, w(1.0) ~ 1.0]
    domains = [x ∈ Interval(1.0, 2.0)]

    chains = [Chain(Dense(1, 15, σ), Dense(15, 1)) for _ in 1:2]
    strategy = GridTraining(0.1)
    discretization = PhysicsInformedNN(chains, strategy)
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x), w(x)])
    prob = discretize(pde_system, discretization)

    res = solve(prob, BFGS(); callback, maxiters = 200)
    xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
    phi = discretization.phi
    u_predict = [(phi[1]([x], res.u.depvar.u))[1] for x in xs]
    w_predict = [(phi[2]([x], res.u.depvar.w))[1] for x in xs]
    u_real = [x for x in xs]
    w_real = [1 / x^2 for x in xs]

    @test mean(abs2, u_real .- u_predict) < 0.001
    @test mean(abs2, w_real .- w_predict) < 0.001
end
