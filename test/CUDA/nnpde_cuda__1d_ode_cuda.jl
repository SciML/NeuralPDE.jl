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

@testset "1D ODE - CUDA" begin
    using Lux, Optimization, OptimizationOptimisers, Random, ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~ θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0f0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    # Discretization
    dt = 0.1f0

    # Neural network
    inner = 20
    chain = Chain(
        Dense(1, inner, σ), Dense(inner, inner, σ), Dense(inner, inner, σ),
        Dense(inner, inner, σ), Dense(inner, inner, σ), Dense(inner, 1)
    )

    strategy = GridTraining(dt)
    ps = Lux.initialparameters(Random.default_rng(), chain) |> ComponentArray |> gpud
    discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u(θ)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(1.0e-2); maxiters = 2000)
    phi = discretization.phi
    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = [infimum(d.domain):(dt / 10):supremum(d.domain) for d in domains][1]

    u_real = [analytic_sol_func(t) for t in ts]
    u_predict = [first(Array(phi([t], res.u))) for t in ts]
    @test u_predict ≈ u_real atol = 0.2
end
