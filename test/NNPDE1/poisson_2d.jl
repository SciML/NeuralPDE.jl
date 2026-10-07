include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

@testset "2D Poisson" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    analytic(x, y) = sinpi(x) * sinpi(y) / (2pi^2)
    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))

    @testset "$(nameof(typeof(strategy)))" for strategy in pde_strategies()
        disc = PhysicsInformedNN(chain, strategy; rng = Xoshiro(1))
        sys = symbolic_discretize(pde_system, disc)
        @test length(unknowns(sys)) == 1
        @test length(ModelingToolkit.get_costs(sys)) == 5
        prob = discretize(pde_system, disc)
        @test prob isa OptimizationProblem
        cb = resampling_callback(prob)
        sol = train(prob; callback = cb)
        @test sol isa PDENoTimeSolution
        @test sol.original_sol.objective < 1.0e-3
        xs = ys = 0:0.05:1
        u_predict = [sol(xi, yi; dv = u(x, y)) for xi in xs, yi in ys]
        u_real = [analytic(xi, yi) for xi in xs, yi in ys]
        @test maximum(abs, u_predict .- u_real) < 0.01
    end

    @testset "solution interface" begin
        disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(2), eval_points = 21)
        prob = discretize(pde_system, disc)
        sol = train(prob)
        @test sol[x] == collect(range(0.0, 1.0; length = 21))
        @test size(sol[u(x, y)]) == (21, 21)
        @test sol[u(x, y)][3, 5] ≈ sol(sol[x][3], sol[y][5]; dv = u(x, y))
        @test sol[u(x, y), 3, 5] == sol[u(x, y)][3, 5]
        ugrid = sol(0:0.5:1, 0:0.25:1; dv = u(x, y))
        @test size(ugrid) == (3, 5)
        @test ugrid[2, 3] ≈ sol(0.5, 0.5; dv = u(x, y))
        @test only(sol(0.5, 0.5)) ≈ sol(0.5, 0.5; dv = u(x, y))
        @test sol(:, :; dv = u(x, y)) == sol[u(x, y)]
        @test sol.original_sol isa SciMLBase.OptimizationSolution
        @test sol.retcode == sol.original_sol.retcode
        @test_throws ErrorException sol[Symbolics.Num(1)]
        @test SciMLBase.problem_type(prob) === pinn_metadata(prob)
    end
end
