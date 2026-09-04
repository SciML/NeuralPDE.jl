using CUDA, ComponentArrays, Lux, LuxCUDA, ModelingToolkit, NeuralPDE, Random, Test
using DomainSets: Interval

@testset "QuadratureTraining with Float64 CUDA parameters" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    equation = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
    boundary_conditions = [
        u(0, y) ~ 0.0,
        u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0,
        u(x, 1) ~ 0.0,
    ]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    chain = Chain(
        Dense(2 => 18, tanh),
        Dense(18 => 18, tanh),
        Dense(18 => 1)
    )
    parameters = Lux.initialparameters(Random.default_rng(), chain) |>
        ComponentArray |> gpu_device() .|> Float64
    discretization = PhysicsInformedNN(
        chain, QuadratureTraining(); init_params = parameters
    )
    @named pde_system = PDESystem(
        equation, boundary_conditions, domains, [x, y], [u(x, y)]
    )
    problem = discretize(pde_system, discretization)
    loss = problem.f(problem.u0, problem.p)
    CUDA.synchronize()

    @test isfinite(loss)
end
