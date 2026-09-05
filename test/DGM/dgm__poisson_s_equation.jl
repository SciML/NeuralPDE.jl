using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "Poisson's equation" begin
    using ComponentArrays, Distributions, Lux, ModelingToolkit, Optimization,
        OptimizationOptimisers, LinearAlgebra, StableRNGs
    import DomainSets: Interval, infimum, supremum
    import QuasiMonteCarlo: LatinHypercubeSample

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    # Initial and boundary conditions
    bcs = [
        u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
        u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1),
    ]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    rng = StableRNG(18)
    strategy = QuasiRandomTraining(
        256; sampling_alg = LatinHypercubeSample(rng), resampling = false, minibatch = 1
    )
    chain = NeuralPDE.DGM(2, 1, 20, 3, tanh, tanh, identity)
    init_params, init_states = Lux.setup(rng, chain)
    discretization = PhysicsInformedNN(
        chain, strategy; init_params = ComponentArray{Float64}(init_params), init_states
    )

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)

    callback = function (p, l)
        p.iter % 50 == 0 && println("$(p.iter) => $l")
        return false
    end

    res = solve(prob, Adam(0.01); callback, maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.001); callback, maxiters = 200)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
    u_real = [analytic_sol_func(x, y) for x in xs for y in ys]

    rmse_scale = sqrt(length(u_real))
    rmse_limit = 0.4 / rmse_scale
    @test norm(u_real) / rmse_scale > rmse_limit
    @test norm(u_real - u_predict) / rmse_scale ≤ rmse_limit
end
