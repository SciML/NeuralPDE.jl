include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

@testset "scaled argument against manufactured solution" begin
    # Exact solution u(x) = x^2 satisfies Dx(u(2x)) = 8x and u(2x) = 4 u(x).
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    eqs = [Dx(u(2x)) ~ 8x, u(2x) ~ 4 * u(x)]
    bcs = [u(0.0) ~ 0.0, u(1.0) ~ 1.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
    chain = Chain(Dense(1, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(7))
    md = pinn_metadata(symbolic_discretize(pde_system, disc))
    @test length(md.blocks) == 4
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 1500, bfgs_iters = 2000)
    xs = 0:0.05:1
    @test maximum(abs, [sol(xi; dv = u(x)) - xi^2 for xi in xs]) < 0.05
end

@testset "delay condition against manufactured solution" begin
    # Exact solution u(t, x) = sinpi(2 * (x - t)) is 1-periodic in x, so
    # u(t, x + 1) = u(t, x), and satisfies the advection equation.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) + Dx(u(t, x)) ~ 0
    bcs = [u(t, x + 1) ~ u(t, x), u(0, x) ~ sinpi(2x)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    chain = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(13))
    md = pinn_metadata(symbolic_discretize(pde_system, disc))
    delay = md.blocks[2]
    @test isequal(delay.ivs, [t, x])
    @test occursin("nn_vcat", string(delay.residual))
    # IC `u(0, x)` still pins `t` for GridTraining despite the delay call.
    pde_block = md.blocks[1]
    @test 0.0 in pde_block.pinned[1]
    @test isempty(pde_block.pinned[2])
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 1000, bfgs_iters = 2000)
    analytic(t, x) = sinpi(2 * (x - t))
    ts = 0:0.1:0.5
    xs = 0:0.1:1
    @test maximum(abs, [sol(ti, xi; dv = u(t, x)) - analytic(ti, xi) for ti in ts, xi in xs]) < 0.1
    @test maximum(abs, [sol(ti, xi; dv = u(t, x)) - sol(ti, xi + 1; dv = u(t, x)) for ti in ts, xi in xs]) < 0.05
end

@testset "reflection argument against manufactured solution" begin
    # Exact solution u(x) = cospi(2x) satisfies u(1 - x) = u(x) and
    # Dx(u) = -2π sinpi(2x); u(0) = 1 rules out the trivial zero solution of the
    # corresponding eigenproblem.
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    eq = Dx(u(x)) ~ -2 * π * sinpi(2x)
    bcs = [u(1 - x) ~ u(x), u(0.0) ~ 1.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    chain = Chain(Dense(1, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(17))
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 1500, bfgs_iters = 2000)
    xs = 0:0.05:1
    @test maximum(abs, [sol(xi; dv = u(x)) - cospi(2xi) for xi in xs]) < 0.08
    @test maximum(abs, [sol(1 - xi; dv = u(x)) - sol(xi; dv = u(x)) for xi in xs]) < 0.05
end
