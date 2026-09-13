include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

@testset "3rd-order ODE with mixed boundary conditions" begin
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    Dxxx = Differential(x)^3
    eq = Dxxx(u(x)) ~ cospi(x)
    bcs = [u(0.0) ~ 0.0, u(1.0) ~ cospi(1.0), Dx(u(1.0)) ~ 1.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    chain = Chain(Dense(1, 8, σ), Dense(8, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(3))
    md = pinn_metadata(symbolic_discretize(pde_system, disc))
    # `Dx(u(1.0))` is a point condition: no free coordinate, so no collocation parameter.
    @test md.blocks[end].xs === nothing
    @test md.blocks[end].npoints == 1
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 300, bfgs_iters = 1000)
    analytic(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sinpi(x)) / (π^3)
    xs = 0:0.05:1
    @test maximum(abs, [sol(xi; dv = u(x)) for xi in xs] .- analytic.(xs)) < 0.02
end

@testset "wave equation with a second time derivative" begin
    @parameters t x
    @variables u(..)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2
    C = 1.0
    eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))
    bcs = [u(t, 0) ~ 0.0, u(t, 1) ~ 0.0, u(0, x) ~ x * (1.0 - x), Differential(t)(u(0, x)) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(4))
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 500, bfgs_iters = 1500)
    analytic(t, x) = sum(
        8 / (k^3 * pi^3) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50
    )
    ts = xs = 0:0.1:1
    @test maximum(abs, [sol(ti, xi; dv = u(t, x)) - analytic(ti, xi) for ti in ts, xi in xs]) < 0.05
end

@testset "mixed derivative" begin
    @parameters x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    # u = x * y is the unique solution: u_xy = 1 with u(x, 0) = u(0, y) = 0.
    eq = Dx(Dy(u(x, y))) + Dy(Dx(u(x, y))) ~ 2
    bcs = [u(x, 0) ~ 0.0, u(0, y) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(5))
    sys = symbolic_discretize(pde_system, disc)
    # `Dx(Dy(u))` and `Dy(Dx(u))` both lower to nested first-order stencils.
    @test length(ModelingToolkit.get_costs(sys)) == 3
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 500, bfgs_iters = 1500)
    xs = ys = 0:0.1:1
    @test maximum(abs, [sol(xi, yi; dv = u(x, y)) - xi * yi for xi in xs, yi in ys]) < 0.05
end
