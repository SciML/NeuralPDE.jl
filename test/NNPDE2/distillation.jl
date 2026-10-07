include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))

using ComponentArrays: ComponentArray, getaxes
using ADTypes: AutoZygote

# `train` in pinn_setup.jl unwraps `PDENoTimeSolution`s; a distilled problem solves
# to a plain `OptimizationSolution`, so warm-start from `res.u` directly.
function train_flat(prob; adam_iters = 500, bfgs_iters = 500)
    res = solve(prob, Adam(0.01); maxiters = adam_iters)
    prob = remake(prob; u0 = res.u)
    return solve(prob, BFGS(; linesearch = BackTracking()); maxiters = bfgs_iters)
end

# Rebuild the parameter container of a distilled student network from the flat
# vector returned by `solve(distill(...))`. `Lux.setup` is only used for the
# container structure; the values come from the trained vector. Fresh state is
# exact because `distill` rejects stateful students.
function student_params(chain, θflat)
    template, st = Lux.setup(Xoshiro(0), chain)
    return ComponentArray(θflat, getaxes(ComponentArray(template))), st
end

# A Lux layer carrying data in its state; `distill` must reject it because the
# training objective and the documented evaluation both use the setup state.
struct OffsetStudent <: Lux.AbstractLuxLayer end
Lux.initialparameters(::AbstractRNG, ::OffsetStudent) =
    (weight = zeros(1, 2), bias = zeros(1))
Lux.initialstates(rng::AbstractRNG, ::OffsetStudent) = (offset = rand(rng),)
(::OffsetStudent)(x, ps, st) = (ps.weight * x .+ ps.bias .+ st.offset, st)

@testset "transfer learning by distillation" begin
    @parameters x y
    @variables u(..) v(..)
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

    # Accuracy budget for the fidelity checks below: the analytic solution ranges
    # over [0, amp], so any constant student is at least amp / 2 from it, and the
    # first-teacher mutation measured 0.0164 on the right subdomain, while the
    # trained students below land near 0.001. A budget of 0.2 * amp therefore
    # rejects both controls with margin.
    amp = maximum(u_real)

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
        @test maximum(abs, u_student .- u_teacher) < 0.2 * amp
        @test maximum(abs, u_student .- u_real) < 0.2 * amp
        # Held-out grid: offset from the evaluation grid above and almost surely
        # disjoint from the random training points and the teacher collocation.
        xsh = 0.025:0.05:1.0
        u_hold = [
            only(first(student_chain(reshape(Float64[xi, yi], 2, 1), θ, st)))
                for xi in xsh, yi in xsh
        ]
        t_hold = [teacher(xi, yi; dv = u(x, y)) for xi in xsh, yi in xsh]
        @test maximum(abs, u_hold .- t_hold) < 0.2 * amp
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
        # The decomposition student must track each subdomain teacher, not just
        # the analytic solution: held-out interior points of each subdomain.
        xl = 0.025:0.05:0.5
        xr = 0.525:0.05:1.0
        xh = 0.025:0.05:1.0
        g_left = [
            only(first(global_chain(reshape(Float64[xi, yi], 2, 1), θ, st)))
                for xi in xl, yi in xh
        ]
        t_left = [left_sol(xi, yi; dv = u(x, y)) for xi in xl, yi in xh]
        @test maximum(abs, g_left .- t_left) < 0.2 * amp
        g_right = [
            only(first(global_chain(reshape(Float64[xi, yi], 2, 1), θ, st)))
                for xi in xr, yi in xh
        ]
        t_right = [right_sol(xi, yi; dv = u(x, y)) for xi in xr, yi in xh]
        @test maximum(abs, g_right .- t_right) < 0.2 * amp
    end

    @testset "argument order and deterministic targets" begin
        Dx = Differential(x)
        odomains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
        @named order_system = PDESystem(
            [Dx(u(x, y)) ~ 1, Dx(v(y, x)) ~ 2], [u(0, y) ~ 10y, v(y, 0) ~ 20y],
            odomains, [x, y], [u(x, y), v(y, x)]
        )
        u_init = (weight = reshape([1.0, 10.0], 1, 2), bias = [0.0])
        v_init = (weight = reshape([20.0, 2.0], 1, 2), bias = [0.0])
        oprob = discretize(
            order_system,
            PhysicsInformedNN(
                [Dense(2, 1), Dense(2, 1)], GridTraining(0.5);
                init_params = [u_init, v_init], rng = Xoshiro(7)
            ),
        )
        # The callback aborts before the first iteration, so the teacher is
        # exactly the prescribed affine network: u(x, y) = x + 10y and, in
        # physical coordinates, v(x, y) = 2x + 20y.
        oteacher = solve(oprob, BFGS(); maxiters = 1, callback = (s, l) -> true)
        X = [0.1 0.2 0.4 0.8; 0.8 0.3 0.6 0.1]
        @test [oteacher(X[1, j], X[2, j]; dv = u(x, y)) for j in axes(X, 2)] ≈
            [X[1, j] + 10X[2, j] for j in axes(X, 2)]
        correct = (weight = [1.0 10.0; 2.0 20.0], bias = [0.0, 0.0])
        dcorrect = distill(oteacher, Dense(2, 2); points = X, init_params = correct)
        @test dcorrect isa OptimizationProblem
        @test dcorrect.f(dcorrect.u0, dcorrect.p) < 1.0e-20
        # The documented evaluation recipe reproduces the fitted function: for
        # stateless students the fresh setup state equals the training state.
        ostudent = Dense(2, 2)
        tpl, rst = Lux.setup(Xoshiro(0), ostudent)
        oθ = ComponentArray(dcorrect.u0, getaxes(ComponentArray(tpl)))
        targets = [X[1:1, :] .+ 10 .* X[2:2, :]; 2 .* X[1:1, :] .+ 20 .* X[2:2, :]]
        @test maximum(abs, first(ostudent(X, oθ, rst)) .- targets) < 1.0e-12
        # A student with v's coordinates swapped misses the targets by 18(x - y).
        swapped = (weight = [1.0 10.0; 20.0 2.0], bias = [0.0, 0.0])
        dswapped = distill(oteacher, Dense(2, 2); points = X, init_params = swapped)
        @test dswapped.f(dswapped.u0, dswapped.p) > 10.0
        # Single reordered variable through the separate-network path. The student
        # always sees inputs in independent-variable order, so its weights are
        # the physical map v(x, y) = 2x + 20y, unlike the teacher network order.
        v_physical = (weight = reshape([2.0, 20.0], 1, 2), bias = [0.0])
        vonly = distill(
            oteacher, Dense(2, 1); points = X, dvs = v(y, x), init_params = v_physical
        )
        @test vonly.f(vonly.u0, vonly.p) < 1.0e-20
        # Tuples select variables like arrays do.
        dtuple = distill(
            oteacher, Dense(2, 2); points = X, dvs = (u(x, y), v(y, x)),
            init_params = correct,
        )
        @test dtuple.f(dtuple.u0, dtuple.p) < 1.0e-20
        # A stateful student is rejected instead of silently mistrained.
        @test_throws ArgumentError distill(oteacher, OffsetStudent(); points = X, rng = Xoshiro(9))
    end

    @testset "overlapping teachers compromise by density" begin
        Dx = Differential(x)
        odomains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
        @named order_system = PDESystem(
            [Dx(u(x, y)) ~ 1, Dx(v(y, x)) ~ 2], [u(0, y) ~ 10y, v(y, 0) ~ 20y],
            odomains, [x, y], [u(x, y), v(y, x)]
        )
        u_init = (weight = reshape([1.0, 10.0], 1, 2), bias = [0.0])
        v_init = (weight = reshape([20.0, 2.0], 1, 2), bias = [0.0])
        oprob = discretize(
            order_system,
            PhysicsInformedNN(
                [Dense(2, 1), Dense(2, 1)], GridTraining(0.5);
                init_params = [u_init, v_init], rng = Xoshiro(7)
            ),
        )
        oteacher = solve(oprob, BFGS(); maxiters = 1, callback = (s, l) -> true)
        # A second teacher shifted by exactly 2 in u: duplicated points must fit
        # their mean, so a mean-initialized student has objective exactly 1.
        oteacher2 = solve(
            remake(oprob; u0 = oprob.u0 .+ [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]), BFGS();
            maxiters = 1, callback = (s, l) -> true,
        )
        X = [0.1 0.2 0.4 0.8; 0.8 0.3 0.6 0.1]
        @test oteacher2(0.2, 0.3; dv = u(x, y)) - oteacher(0.2, 0.3; dv = u(x, y)) ≈ 2.0
        mean_init = (weight = reshape([1.0, 10.0], 1, 2), bias = [1.0])
        doverlap = distill(
            [oteacher, oteacher2], Dense(2, 1); points = [X, X], dvs = u(x, y),
            init_params = mean_init,
        )
        @test abs(doverlap.f(doverlap.u0, doverlap.p) - 1.0) < 1.0e-10
        # A student initialized to the first teacher exactly has half its
        # residuals vanish and half equal 2^2; a mutation using only the first
        # teacher's targets drives this objective to 0 instead of 2.
        dt1 = distill(
            [oteacher, oteacher2], Dense(2, 1); points = [X, X], dvs = u(x, y),
            init_params = u_init,
        )
        @test abs(dt1.f(dt1.u0, dt1.p) - 2.0) < 1.0e-10
    end

    @testset "teacher coordinate validation" begin
        @parameters t
        @variables w(..)
        Dt = Differential(t)
        @named tiny_system = PDESystem(
            Dt(w(t)) ~ 1.0, [w(0.0) ~ 0.0], [t ∈ Interval(0.0, 1.0)], [t], [w(t)]
        )
        # Explicit non-Enzyme adtype: the default AutoEnzyme hits the tracked
        # runtime-activity failure (#1194) on this fixture before the abort callback.
        tiny_prob = discretize(
            tiny_system, PhysicsInformedNN(Dense(1, 1), GridTraining(0.5); rng = Xoshiro(8));
            adtype = AutoZygote(),
        )
        tiny_teacher = solve(tiny_prob, BFGS(); maxiters = 1, callback = (s, l) -> true)
        other_chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
        @test_throws ArgumentError distill(
            [teacher, tiny_teacher], other_chain; npoints = 5, rng = Xoshiro(9)
        )
        @test_throws ArgumentError distill(
            tiny_teacher, Dense(1, 2); dvs = [w(t), u(x, y)], npoints = 5,
            rng = Xoshiro(9),
        )
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
