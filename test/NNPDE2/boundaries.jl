include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

@testset "periodic boundary condition (issue #469)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) + Dx(u(t, x)) ~ 0
    bcs = [u(t, 0) ~ u(t, 1), u(0, x) ~ sinpi(2x)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    chain = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(12))
    md = pinn_metadata(symbolic_discretize(pde_system, disc))
    periodic = md.blocks[2]
    @test isequal(periodic.ivs, [t])
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 1000, bfgs_iters = 2000)
    analytic(t, x) = sinpi(2 * (x - t))
    ts = 0:0.1:0.5
    xs = 0:0.1:1
    @test maximum(abs, [sol(ti, xi; dv = u(t, x)) - analytic(ti, xi) for ti in ts, xi in xs]) < 0.1
    @test maximum(abs, [sol(ti, 0.0; dv = u(t, x)) - sol(ti, 1.0; dv = u(t, x)) for ti in ts]) < 0.05
end

@testset "boundary conditions as constraints" begin
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    eq = Dx(u(x)) ~ -u(x)
    bcs = [u(0.0) ~ 1.0, u(1.0) ~ exp(-1.0)]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    chain = Chain(Dense(1, 8, σ), Dense(8, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1); boundary_policy = :constraints)
    sys = symbolic_discretize(pde_system, disc)
    @test length(ModelingToolkit.get_costs(sys)) == 1
    @test length(ModelingToolkit.constraints(sys)) == 2
    @test_throws ArgumentError PhysicsInformedNN(chain, GridTraining(0.1); boundary_policy = :foo)
end

@testset "unsupported constructs give clear errors" begin
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    domains = [x ∈ Interval(0.0, 1.0)]
    chain = Chain(Dense(1, 4, σ), Dense(4, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1))
    Ix = Integral(x in DomainSets.ClosedInterval(0.0, 1.0))
    @named integral_system = PDESystem([Ix(u(x)) ~ 1.0], [u(0.0) ~ 0.0], domains, [x], [u(x)])
    @test_throws ArgumentError symbolic_discretize(integral_system, disc)
    @named shifted_system = PDESystem([u(2x) ~ 1.0], [u(0.0) ~ 0.0], domains, [x], [u(x)])
    # PDEBase's variable map rejects the shifted argument before the lowering does.
    @test_throws Exception symbolic_discretize(shifted_system, disc)
    @test_throws ArgumentError PhysicsInformedNN(chain, WeightedIntervalTraining([1.0], 10)) |>
        d -> symbolic_discretize(integral_system, d)
end
