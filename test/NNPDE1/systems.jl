include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

@testset "system of ODEs with one network per variable and a shared network" begin
    @parameters x
    @variables u1(..) u2(..)
    Dx = Differential(x)
    eqs = [Dx(u1(x)) ~ u2(x), Dx(u2(x)) ~ -u1(x)]
    bcs = [u1(0.0) ~ 0.0, u2(0.0) ~ 1.0]
    domains = [x ∈ Interval(0.0, pi / 2)]
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u1(x), u2(x)])
    xs = range(0.0, pi / 2; length = 21)

    @testset "one chain per dependent variable" begin
        chains = [Chain(Dense(1, 8, σ), Dense(8, 1)) for _ in 1:2]
        disc = PhysicsInformedNN(chains, GridTraining(0.05); rng = Xoshiro(6))
        sys = symbolic_discretize(pde_system, disc)
        @test length(unknowns(sys)) == 2
        prob = discretize(pde_system, disc)
        sol = train(prob; adam_iters = 300, bfgs_iters = 1000)
        @test maximum(abs, [sol(xi; dv = u1(x)) for xi in xs] .- sin.(xs)) < 0.02
        @test maximum(abs, [sol(xi; dv = u2(x)) for xi in xs] .- cos.(xs)) < 0.02
    end

    @testset "shared chain with two outputs" begin
        chain = Chain(Dense(1, 8, σ), Dense(8, 2))
        disc = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(7))
        sys = symbolic_discretize(pde_system, disc)
        @test length(unknowns(sys)) == 1
        md = pinn_metadata(sys)
        @test md.networks[1].output == 1 && md.networks[2].output == 2
        prob = discretize(pde_system, disc)
        sol = train(prob; adam_iters = 300, bfgs_iters = 1000)
        @test maximum(abs, [sol(xi; dv = u1(x)) for xi in xs] .- sin.(xs)) < 0.02
        @test maximum(abs, [sol(xi; dv = u2(x)) for xi in xs] .- cos.(xs)) < 0.02
    end
end

@testset "heterogeneous system: dependent variables with different arguments" begin
    @parameters x y
    @variables p(..) q(..) r(..) s(..)
    Dx = Differential(x)
    Dy = Differential(y)
    # p = x^2 + y^2, q = x^2 - y^2, r = 2x, s = 2y
    eqs = [
        p(x, y) + q(x, y) ~ 2x^2,
        p(x, y) - q(x, y) ~ 2y^2,
        Dx(r(x)) ~ 2, Dy(s(y)) ~ 2,
    ]
    bcs = [
        p(0, y) ~ y^2, q(x, 0) ~ x^2, r(0) ~ 0.0, s(0) ~ 0.0,
    ]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eqs, bcs, domains, [x, y], [p(x, y), q(x, y), r(x), s(y)])
    chains = [
        Chain(Dense(2, 10, σ), Dense(10, 1)), Chain(Dense(2, 10, σ), Dense(10, 1)),
        Chain(Dense(1, 6, σ), Dense(6, 1)), Chain(Dense(1, 6, σ), Dense(6, 1)),
    ]
    disc = PhysicsInformedNN(chains, GridTraining(0.1); rng = Xoshiro(8))
    md = pinn_metadata(symbolic_discretize(pde_system, disc))
    @test [length(b.ivs) for b in md.blocks] == [2, 2, 1, 1, 1, 1, 0, 0]
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 500, bfgs_iters = 1500)
    xs = ys = 0:0.25:1
    @test maximum(abs, [sol(xi, yi; dv = p(x, y)) - (xi^2 + yi^2) for xi in xs, yi in ys]) < 0.05
    @test maximum(abs, [sol(xi, yi; dv = q(x, y)) - (xi^2 - yi^2) for xi in xs, yi in ys]) < 0.05
    @test maximum(abs, [sol(xi; dv = r(x)) - 2xi for xi in xs]) < 0.05
    @test maximum(abs, [sol(yi; dv = s(y)) - 2yi for yi in ys]) < 0.05
end
