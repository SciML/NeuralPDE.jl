include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using SymbolicIndexingInterface: getp

@testset "Latin hypercube collocation" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0.0
    bcs = [u(-2, y) ~ 0.0, u(3, y) ~ 0.0, u(x, 1) ~ 0.0, u(x, 5) ~ 0.0]
    domains = [x ∈ Interval(-2.0, 3.0), y ∈ Interval(1.0, 5.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    chain = Chain(Dense(2, 4, σ), Dense(4, 1))

    function check_latin_hypercube(X, lower, upper, n, T)
        @test size(X) == (length(lower), n)
        @test eltype(X) == T
        for i in eachindex(lower)
            @test all(lower[i] .<= X[i, :] .< upper[i])
            strata = floor.(Int, n .* (X[i, :] .- lower[i]) ./ (upper[i] - lower[i]))
            @test sort(strata) == 0:(n - 1)
        end
    end

    for refresh in (false, true)
        strategy = QuasiRandomTraining(
            16; bcs_points = 8, sampling_alg = LatinHypercubeSample(Xoshiro(17)),
            resampling = refresh
        )
        prob = discretize(pdesys, PhysicsInformedNN(chain, strategy; rng = Xoshiro(18)))
        blocks = pinn_metadata(prob).blocks
        @test length(blocks) == 5
        coordinates() = [copy(getp(prob, block.xs)(prob)) for block in blocks]
        before = coordinates()
        resample!(prob)
        after = coordinates()
        for draws in (before, after)
            check_latin_hypercube(draws[1], [-2.0, 1.0], [3.0, 5.0], 16, Float64)
            for i in 2:5
                lo, hi = i <= 3 ? ([1.0], [5.0]) : ([-2.0], [3.0])
                check_latin_hypercube(draws[i], lo, hi, 8, Float64)
            end
        end
        @test all((a != b) == refresh for (a, b) in zip(before, after))
        if refresh
            @test Set(Tuple.(eachcol(before[1]))) != Set(Tuple.(eachcol(after[1])))
        end
    end

    lower, upper = Float32[-2, 1], Float32[3, 5]
    matrices = QuasiMonteCarlo.generate_design_matrices(16, lower, upper, LatinHypercubeSample(Xoshiro(19)), 4)
    @test length(matrices) == 4
    for matrix in matrices
        check_latin_hypercube(matrix, lower, upper, 16, Float32)
    end
    @test length(unique(matrices)) == 4

    function batch_trace(seed)
        seen = AbstractMatrix[]
        strategy = QuasiRandomTraining(
            16; sampling_alg = LatinHypercubeSample(Xoshiro(19)), resampling = false, minibatch = 4
        )
        loss = NeuralPDE.get_loss_function(
            Float32[0], (X, θ) -> (push!(seen, copy(X)); X), (lower, upper), Float32, strategy
        )
        @sync @async begin
            Random.seed!(seed)
            for _ in 1:64
                loss(Float32[0])
            end
        end
        @test all(X -> eltype(X) === Float32, seen)
        return seen
    end
    seen = batch_trace(20)
    @test seen == batch_trace(20)
    @test seen != batch_trace(21)
    batches = unique(seen)
    @test length(batches) == 4
    for matrix in batches
        check_latin_hypercube(matrix, lower, upper, 16, Float32)
    end
end

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
