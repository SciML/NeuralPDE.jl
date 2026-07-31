module CUDATestSetup

    using LuxCUDA, Lux

    function callback(p, l)
        if p.iter == 1 || p.iter % 250 == 0
            println("Current loss is: $l after $(p.iter) iterations")
        end
        return false
    end

    const gpud = gpu_device()

    export gpud, callback

end

using .CUDATestSetup

using NeuralPDE
using Test

@testset "1D PDE Dirichlet BC - CUDA" begin
    using Lux, Optimization, OptimizationOptimisers, Random, ComponentArrays
    import DomainSets: Interval, infimum, supremum
    import Boltz.Layers: PeriodicEmbedding

    Random.seed!(100)

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, 2π) ~ exp(-t),
    ]

    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 2π)]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    inner = 30
    chain = Chain(
        PeriodicEmbedding([2], [2π]), Dense(3, inner, σ), Dense(inner, inner, σ),
        Dense(inner, inner, σ), Dense(inner, inner, σ),
        Dense(inner, inner, σ), Dense(inner, inner, σ), Dense(inner, 1)
    )

    strategy = StochasticTraining(1000)
    ps, st = Lux.setup(Random.default_rng(), chain)
    ps = ps |> ComponentArray |> gpu_device() |> f64
    st = st |> gpu_device() |> f64

    discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)
    prob = discretize(pdesys, discretization)
    res = solve(prob, Adam(0.01); maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.001); maxiters = 1000)
    phi = discretization.phi
    u_exact = (t, x) -> exp.(-t) * cos.(x)
    ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

    u_predict = [first(Array(phi([t, x], res.u))) for t in ts for x in xs]
    u_real = [u_exact(t, x) for t in ts for x in xs]
    diff_u = abs.(u_predict .- u_real)

    @test u_predict ≈ u_real atol = 1.0
end
