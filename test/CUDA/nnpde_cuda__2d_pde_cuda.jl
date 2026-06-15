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

@testset "2D PDE - CUDA" begin
    using Lux, Optimization, OptimizationOptimisers, Random, ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    t_min, t_max, x_min, x_max, y_min, y_max = 0.0, 2.0, 0.0, 2.0, 0.0, 2.0

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

    # Initial and boundary conditions
    bcs = [
        u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max),
    ]

    # Space and time domains
    domains = [
        t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
    ]

    # Neural network
    inner = 25
    chain = Chain(
        Dense(3, inner, σ), Dense(inner, inner, σ),
        Dense(inner, inner, σ), Dense(inner, inner, σ), Dense(inner, 1)
    )

    strategy = GridTraining(0.05)
    ps = Lux.initialparameters(Random.default_rng(), chain) |> ComponentArray |> gpud |> f64

    discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
    @named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(0.01); maxiters = 2500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, Adam(0.001); maxiters = 2500)
    phi = discretization.phi
    ts, xs, ys = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

    u_real = [analytic_sol_func(t, x, y) for t in ts for x in xs for y in ys]
    u_predict = [
        first(Array(phi([t, x, y], res.u))) for t in ts for x in xs
            for y in ys
    ]

    @test u_predict ≈ u_real rtol = 0.2
end
