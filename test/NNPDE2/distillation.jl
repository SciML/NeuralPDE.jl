include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

using ComponentArrays: ComponentArray, getaxes

# `train` in pinn_setup.jl unwraps `PDENoTimeSolution`s; a distilled problem solves
# to a plain `OptimizationSolution`, so warm-start from `res.u` directly.
function train_flat(prob; adam_iters = 500, bfgs_iters = 500)
    res = solve(prob, Adam(0.01); maxiters = adam_iters)
    prob = remake(prob; u0 = res.u)
    return solve(prob, BFGS(; linesearch = BackTracking()); maxiters = bfgs_iters)
end

# Rebuild the parameter container of a distilled student network from the flat
# vector returned by `solve(distill(...))`. `Lux.setup` is only used for the
# container structure; the values come from the trained vector.
function student_params(chain, θflat)
    template, st = Lux.setup(Xoshiro(0), chain)
    return ComponentArray(θflat, getaxes(ComponentArray(template))), st
end

@testset "transfer learning by distillation" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    analytic(x, y) = sinpi(x) * sinpi(y) / (2pi^2)

    teacher_chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))
    disc = PhysicsInformedNN(teacher_chain, GridTraining(0.1); rng = Xoshiro(1))
    prob = discretize(pde_system, disc)
    teacher = train(prob)

    xs = ys = 0:0.05:1
    u_teacher = [teacher(xi, yi; dv = u(x, y)) for xi in xs, yi in ys]
    u_real = [analytic(xi, yi) for xi in xs, yi in ys]
    @test maximum(abs, u_teacher .- u_real) < 0.01

    @testset "same-architecture warm start" begin
        warm = remake(prob; u0 = teacher.original_sol.u)
        @test warm.u0 ≈ teacher.original_sol.u
        continued = solve(warm, BFGS(; linesearch = BackTracking()); maxiters = 100)
        u_warm = [continued(xi, yi; dv = u(x, y)) for xi in xs, yi in ys]
        @test maximum(abs, u_warm .- u_real) < 0.01
    end

    @testset "distillation into a different architecture" begin
        student_chain = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
        dprob = distill(teacher, student_chain; npoints = 1000, rng = Xoshiro(2))
        @test dprob isa OptimizationProblem
        dres = train_flat(dprob)
        θ, st = student_params(student_chain, dres.u)
        u_student = [
            only(first(student_chain(reshape(Float64[xi, yi], 2, 1), θ, st)))
            for xi in xs, yi in ys
        ]
        @test maximum(abs, u_student .- u_teacher) < 0.05
        @test maximum(abs, u_student .- u_real) < 0.05
    end

    @testset "domain decomposition then distillation" begin
        left_domains = [x ∈ Interval(0.0, 0.5), y ∈ Interval(0.0, 1.0)]
        right_domains = [x ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0)]
        left_bcs = [
            u(0, y) ~ 0.0, u(0.5, y) ~ analytic(0.5, y),
            u(x, 0) ~ 0.0, u(x, 1) ~ 0.0,
        ]
        right_bcs = [
            u(0.5, y) ~ analytic(0.5, y), u(1, y) ~ 0.0,
            u(x, 0) ~ 0.0, u(x, 1) ~ 0.0,
        ]
        @named left_system = PDESystem(eq, left_bcs, left_domains, [x, y], [u(x, y)])
        @named right_system = PDESystem(eq, right_bcs, right_domains, [x, y], [u(x, y)])
        subchain() = Chain(Dense(2, 8, σ), Dense(8, 8, σ), Dense(8, 1))
        left_prob = discretize(
            left_system, PhysicsInformedNN(subchain(), GridTraining(0.1); rng = Xoshiro(3))
        )
        right_prob = discretize(
            right_system, PhysicsInformedNN(subchain(), GridTraining(0.1); rng = Xoshiro(4))
        )
        left_sol = train(left_prob)
        right_sol = train(right_prob)

        u_left = [left_sol(xi, yi; dv = u(x, y)) for xi in 0:0.05:0.5, yi in ys]
        u_right = [right_sol(xi, yi; dv = u(x, y)) for xi in 0.55:0.05:1, yi in ys]
        @test maximum(abs, u_left .- [analytic(xi, yi) for xi in 0:0.05:0.5, yi in ys]) < 0.05
        @test maximum(abs, u_right .- [analytic(xi, yi) for xi in 0.55:0.05:1, yi in ys]) < 0.05

        global_chain = Chain(
            Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 16, tanh), Dense(16, 1)
        )
        dprob = distill([left_sol, right_sol], global_chain; npoints = 500, rng = Xoshiro(5))
        @test dprob isa OptimizationProblem
        dres = train_flat(dprob)
        θ, st = student_params(global_chain, dres.u)
        u_global = [
            only(first(global_chain(reshape(Float64[xi, yi], 2, 1), θ, st)))
            for xi in xs, yi in ys
        ]
        @test maximum(abs, u_global .- u_real) < 0.05
    end

    @testset "argument validation" begin
        other_chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
        @test_throws ArgumentError distill(
            teacher, [other_chain, other_chain]; npoints = 10, rng = Xoshiro(6)
        )
        @test_throws ArgumentError distill(teacher, other_chain; npoints = 0, rng = Xoshiro(6))
        bad_points = zeros(3, 10)
        @test_throws ArgumentError distill(
            teacher, other_chain; points = bad_points, rng = Xoshiro(6)
        )
        two_out = Chain(Dense(2, 4, tanh), Dense(4, 2))
        @test_throws ArgumentError distill(teacher, two_out; npoints = 10, rng = Xoshiro(6))
        selective = distill(teacher, other_chain; dvs = u(x, y), npoints = 10, rng = Xoshiro(6))
        @test selective isa OptimizationProblem
        Xs = hcat([[x, y] for x in 0.0:0.5:1.0 for y in 0.0:0.5:1.0]...)
        explicit = distill(teacher, other_chain; points = Xs, rng = Xoshiro(6))
        @test explicit isa OptimizationProblem
    end
end
