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

@testset "1D PDE Neumann BC - CUDA" begin
    using Lux, Optimization, OptimizationOptimisers, Random, QuasiMonteCarlo,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cos(x),
        Dx(u(t, 0)) ~ 0.0,
        Dx(u(t, 1)) ~ -exp(-t) * sin(1.0),
    ]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    inner = 20
    chain = Chain(
        Dense(2, inner, σ), Dense(inner, inner, σ),
        Dense(inner, inner, σ), Dense(inner, inner, σ), Dense(inner, 1)
    )

    strategy = QuasiRandomTraining(
        500; sampling_alg = SobolSample(), resampling = false, minibatch = 30
    )
    ps = Lux.initialparameters(Random.default_rng(), chain) |> ComponentArray |> gpud |> f64

    discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
    prob = discretize(pdesys, discretization)
    res = solve(prob, Adam(0.1); maxiters = 2000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.01); maxiters = 2000)
    phi = discretization.phi
    u_exact = (t, x) -> exp(-t) * cos(x)
    ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

    u_predict = [first(Array(phi([t, x], res.u))) for t in ts for x in xs]
    u_real = [u_exact(t, x) for t in ts for x in xs]
    diff_u = abs.(u_predict .- u_real)

    @test u_predict ≈ u_real atol = 1.0
end
