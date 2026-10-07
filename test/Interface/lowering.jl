include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using ModelingToolkitBase: getdefault
using SymbolicIndexingInterface: getu, getp
using Zygote, ForwardDiff, Enzyme, LinearAlgebra, Statistics, ADTypes, ComponentArrays

@parameters x y
@variables u(..)
Dx = Differential(x)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
chain = Chain(Dense(2, 8, σ), Dense(8, 1))
disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(1))
prob = discretize(pde_system, disc)
md = pinn_metadata(prob)
net = only(md.networks)
apply = getdefault(net.NN)
θ = prob.u0

@testset "default AD for objective paths" begin
    default = AutoEnzyme
    @test prob.f.adtype isa default
    dgm_disc = DeepGalerkin(
        2, 1, 4, 1, tanh, tanh, identity, GridTraining(0.5); rng = Xoshiro(2)
    )
    @test discretize(pde_system, dgm_disc).f.adtype isa default
    @test discretize(pde_system, dgm_disc; adtype = AutoZygote()).f.adtype isa AutoZygote
    loss_disc = let data = "ab"
        PhysicsInformedNN(
            chain, GridTraining(0.5); rng = Xoshiro(4), additional_loss = function (phi, θ, p)
                out = phi.u([0.5 0.25; 0.5 0.25], θ.u)
                buf = similar(out, length(data))
                for i in eachindex(buf)
                    buf[i] = out[i]
                end
                return sum(abs2, buf)
            end
        )
    end
    loss_prob = discretize(pde_system, loss_disc)
    @test loss_prob.f.adtype isa default
    g_fd = ForwardDiff.gradient(w -> loss_prob.f(w, loss_prob.p), loss_prob.u0)
    # Adam's callback sees the gradient at `state.u` before the update is applied.
    g_default = Ref{Vector{Float64}}()
    solve(
        loss_prob, Adam(0.001); maxiters = 1, callback = function (state, l)
            @test state.u == loss_prob.u0
            g_default[] = copy(state.grad)
            return false
        end
    )
    @test g_default[] ≈ g_fd rtol = 1.0e-6
    data = [0.3 0.1]
    capture_disc = PhysicsInformedNN(
        chain, GridTraining(0.5); rng = Xoshiro(5), additional_loss = function (phi, θ, p)
            return sum(abs2, phi.u([0.5 0.25; 0.5 0.25], θ.u) .- data)
        end
    )
    # Selection only: an `additional_loss` that captures arrays keeps the default
    # backend because the closure is not inspected. Integrity of those captures
    # under reverse-mode AD is covered by the testset below.
    @test discretize(pde_system, capture_disc).f.adtype isa default
    @parameters s τ
    @variables v(..)
    integral = Integral(τ in Interval(0.0, s))
    @named integral_sys = PDESystem(
        integral(v(τ)) ~ s^2 / 2, [v(0.0) ~ 0.0],
        [s ∈ Interval(0.0, 1.0)], [s], [v(s)]
    )
    integral_disc = PhysicsInformedNN(
        Chain(Dense(1, 1)), GridTraining(0.5); rng = Xoshiro(3)
    )
    @test discretize(integral_sys, integral_disc).f.adtype isa AutoZygote
    @test discretize(integral_sys, integral_disc; adtype = AutoEnzyme()).f.adtype isa AutoEnzyme
end

@testset "lowered residual matches a manual finite-difference computation" begin
    block = md.blocks[1]
    X = getp(prob, block.xs)(prob)
    @test size(X) == (2, 81)
    f(X) = apply(X, θ)
    ε = eps(Float64)^(1 / 4)
    uxx = (f(X .+ [ε, 0.0]) .+ f(X .- [ε, 0.0]) .- 2 .* f(X)) ./ ε^2
    uyy = (f(X .+ [0.0, ε]) .+ f(X .- [0.0, ε]) .- 2 .* f(X)) ./ ε^2
    manual = uxx .+ uyy .+ sinpi.(X[1:1, :]) .* sinpi.(X[2:2, :])
    @test getu(prob, block.residual)(prob) ≈ manual
    bc = md.blocks[2]
    Xb = getp(prob, bc.xs)(prob)
    @test size(Xb) == (1, 11)
    @test getu(prob, bc.residual)(prob) ≈ f(vcat(zeros(1, 11), Xb))
    manual_cost = mean(abs2, manual) + sum(
        mean(abs2, getu(prob, b.residual)(prob)) for b in md.blocks[2:end]
    )
    @test prob.f(θ, prob.p) ≈ manual_cost
end

@testset "gradients agree across AD backends" begin
    g_fd = ForwardDiff.gradient(u -> prob.f(u, prob.p), θ)
    g_zy = Zygote.gradient(u -> prob.f(u, prob.p), θ)[1]
    @test g_zy ≈ g_fd rtol = 1.0e-6
    enzyme = AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const
    )
    for adtype in (NeuralPDE.default_adtype(), enzyme, AutoForwardDiff())
        p = discretize(pde_system, disc; adtype)
        @test p.f.adtype === adtype
        res = solve(p, Adam(0.001); maxiters = 50)
        @test res.original_sol.objective < prob.f(θ, prob.p)
    end
end

@testset "Enzyme preserves captured additional-loss data" begin
    captured_xs = [0.2 0.4 0.6 0.8; 0.2 0.4 0.6 0.8]
    captured_ys = zeros(1, 4)
    additional_loss = (phi, θ, p) -> mean(abs2, phi.u(captured_xs, θ.u) .- captured_ys)
    captured_disc = PhysicsInformedNN(
        chain, GridTraining(0.5); rng = Xoshiro(3), additional_loss
    )
    captured_prob = discretize(pde_system, captured_disc)
    captured_xs_before, captured_ys_before = copy(captured_xs), copy(captured_ys)
    restore!() = (captured_xs .= captured_xs_before; captured_ys .= captured_ys_before)
    @test captured_prob.f.adtype === NeuralPDE.default_adtype()
    # the adtype whose reverse pass overwrote captured arrays in #1170
    runtime_enzyme = AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const
    )

    g_fd = ForwardDiff.gradient(θ -> captured_prob.f(θ, captured_prob.p), captured_prob.u0)
    let first_objective = (f, θ, p) -> (v = f(θ, p); v isa AbstractFloat ? v : first(v))
        for mode in (Enzyme.Reverse, Enzyme.set_runtime_activity(Enzyme.Reverse))
            restore!()
            g_enzyme = zero(captured_prob.u0)
            Enzyme.autodiff(
                mode, Enzyme.Const(first_objective), Enzyme.Active,
                Enzyme.Const(captured_prob.f.f),
                Enzyme.Duplicated(captured_prob.u0, g_enzyme), Enzyme.Const(captured_prob.p)
            )
            @test g_enzyme ≈ g_fd rtol = 1.0e-6
            @test captured_xs == captured_xs_before
            @test captured_ys == captured_ys_before
        end
    end

    restore!()
    solve(captured_prob, Adam(0.001); maxiters = 2)
    @test captured_xs == captured_xs_before
    @test captured_ys == captured_ys_before

    restore!()
    runtime_prob = discretize(pde_system, captured_disc; adtype = runtime_enzyme)
    solve(runtime_prob, Adam(0.001); maxiters = 2)
    @test captured_xs == captured_xs_before
    @test captured_ys == captured_ys_before
end

@testset "resampling and remake" begin
    sdisc = PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20); rng = Xoshiro(2))
    sprob = discretize(pde_system, sdisc)
    smd = pinn_metadata(sprob)
    X0 = copy(getp(sprob, smd.blocks[1].xs)(sprob))
    @test size(X0) == (2, 50)
    @test size(getp(sprob, smd.blocks[2].xs)(sprob)) == (1, 20)
    resample!(sprob)
    X1 = getp(sprob, smd.blocks[1].xs)(sprob)
    @test X1 != X0
    @test all(0 .<= X1 .<= 1)
    gdisc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(2))
    gprob = discretize(pde_system, gdisc)
    Xg = copy(getp(gprob, pinn_metadata(gprob).blocks[1].xs)(gprob))
    resample!(gprob)
    @test getp(gprob, pinn_metadata(gprob).blocks[1].xs)(gprob) == Xg
    new_points = rand(Xoshiro(3), 2, 81)
    rprob = remake(prob; p = [md.blocks[1].xs => new_points])
    @test getp(rprob, md.blocks[1].xs)(rprob) == new_points
    @test rprob.f(θ, rprob.p) != prob.f(θ, prob.p)
    # transfer learning: warm start from trained weights
    res = solve(prob, Adam(0.01); maxiters = 20)
    wprob = remake(prob; u0 = res.original_sol.u)
    @test wprob.u0 == res.original_sol.u
end

@testset "discretize is reproducible regardless of a preceding symbolic_discretize" begin
    rdisc = PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20); rng = Xoshiro(7))
    prob_alone = discretize(pde_system, rdisc)

    sdisc = PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20); rng = Xoshiro(7))
    sys = symbolic_discretize(pde_system, sdisc)
    prob_after = discretize(pde_system, sdisc)

    @test prob_after.u0 == prob_alone.u0
    amd = pinn_metadata(prob_alone)
    bmd = pinn_metadata(prob_after)
    @test getp(prob_after, bmd.blocks[1].xs)(prob_after) ==
        getp(prob_alone, amd.blocks[1].xs)(prob_alone)
end

@testset "seeded deterministic resample!" begin
    sdisc = PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20); rng = Xoshiro(2))
    sprob = discretize(pde_system, sdisc)
    smd = pinn_metadata(sprob)
    X0 = copy(getp(sprob, smd.blocks[1].xs)(sprob))
    saved_rng = copy(sdisc.rng)
    p1 = copy(sprob.p)
    p2 = copy(sprob.p)
    resample!(p1, smd; rng = Xoshiro(42))
    resample!(p2, smd; rng = Xoshiro(42))
    @test getp(sprob, smd.blocks[1].xs)(p1) == getp(sprob, smd.blocks[1].xs)(p2)
    @test getp(sprob, smd.blocks[1].xs)(p1) != X0
    # an explicit `rng` leaves the discretization's own stream untouched
    @test sdisc.rng == saved_rng
    @test getp(sprob, smd.blocks[1].xs)(sprob) == X0
end

@testset "seeded resample! drives randomized quasi-random sampling" begin
    qdisc = PhysicsInformedNN(
        chain, QuasiRandomTraining(50; bcs_points = 20, resampling = true);
        rng = Xoshiro(2)
    )
    qprob = discretize(pde_system, qdisc)
    qmd = pinn_metadata(qprob)
    X0 = copy(getp(qprob, qmd.blocks[1].xs)(qprob))
    p1 = copy(qprob.p)
    p2 = copy(qprob.p)
    resample!(p1, qmd; rng = Xoshiro(42))
    resample!(p2, qmd; rng = Xoshiro(42))
    X1 = getp(qprob, qmd.blocks[1].xs)(p1)
    @test X1 == getp(qprob, qmd.blocks[1].xs)(p2)
    @test X1 != X0
    # identically seeded discretizations draw identical initial QMC points
    qdisc2 = PhysicsInformedNN(
        chain, QuasiRandomTraining(50; bcs_points = 20, resampling = true);
        rng = Xoshiro(2)
    )
    qprob2 = discretize(pde_system, qdisc2)
    @test getp(qprob2, pinn_metadata(qprob2).blocks[1].xs)(qprob2) == X0
    # deterministic bases with R = NoRand() ignore `rng` and redraw the same points
    sdisc = PhysicsInformedNN(
        chain,
        QuasiRandomTraining(50; bcs_points = 20, sampling_alg = SobolSample());
        rng = Xoshiro(2)
    )
    sprob = discretize(pde_system, sdisc)
    smd = pinn_metadata(sprob)
    Xs = copy(getp(sprob, smd.blocks[1].xs)(sprob))
    resample!(sprob)
    @test getp(sprob, smd.blocks[1].xs)(sprob) == Xs
    # R-randomized Sobol (power-of-two counts) re-seeds alg.R.rng
    owen = SobolSample(R = OwenScramble(base = 2, pad = 32))
    odisc = PhysicsInformedNN(
        chain,
        QuasiRandomTraining(64; bcs_points = 32, sampling_alg = owen, resampling = true);
        rng = Xoshiro(2)
    )
    oprob = discretize(pde_system, odisc)
    omd = pinn_metadata(oprob)
    Xo0 = copy(getp(oprob, omd.blocks[1].xs)(oprob))
    op1 = copy(oprob.p)
    op2 = copy(oprob.p)
    resample!(op1, omd; rng = Xoshiro(42))
    resample!(op2, omd; rng = Xoshiro(42))
    Xo1 = getp(oprob, omd.blocks[1].xs)(op1)
    @test Xo1 == getp(oprob, omd.blocks[1].xs)(op2)
    @test Xo1 != Xo0
    odisc2 = PhysicsInformedNN(
        chain,
        QuasiRandomTraining(64; bcs_points = 32, sampling_alg = owen, resampling = true);
        rng = Xoshiro(2)
    )
    oprob2 = discretize(pde_system, odisc2)
    @test getp(oprob2, pinn_metadata(oprob2).blocks[1].xs)(oprob2) == Xo0
end

@testset "unseeded discretizations draw distinct weights and points" begin
    d1 = PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20))
    d2 = PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20))
    p1 = discretize(pde_system, d1)
    p2 = discretize(pde_system, d2)
    @test p1.u0 != p2.u0
    @test getp(p1, pinn_metadata(p1).blocks[1].xs)(p1) !=
        getp(p2, pinn_metadata(p2).blocks[1].xs)(p2)
    Random.seed!(11)
    pa = discretize(pde_system, PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20)))
    Random.seed!(11)
    pb = discretize(pde_system, PhysicsInformedNN(chain, StochasticTraining(50; bcs_points = 20)))
    @test pa.u0 == pb.u0
    @test getp(pa, pinn_metadata(pa).blocks[1].xs)(pa) ==
        getp(pb, pinn_metadata(pb).blocks[1].xs)(pb)
end

@testset "non-copyable rng seeds the owned streams once" begin
    ddisc = PhysicsInformedNN(chain, GridTraining(0.1); rng = RandomDevice())
    @test ddisc.rng !== ddisc.init_rng
    @test discretize(pde_system, ddisc).u0 == discretize(pde_system, ddisc).u0
    d2 = PhysicsInformedNN(chain, GridTraining(0.1); rng = RandomDevice())
    @test discretize(pde_system, d2).u0 != discretize(pde_system, ddisc).u0
end

@testset "cost weights and quadrature weights" begin
    wprob = discretize(pde_system, disc; weights = [1.0, 10.0, 10.0, 10.0, 10.0])
    costs = [mean(abs2, getu(prob, b.residual)(prob)) for b in md.blocks]
    @test wprob.f(θ, wprob.p) ≈ costs[1] + 10 * sum(costs[2:end])
    qdisc = PhysicsInformedNN(chain, QuadratureTraining(; quadrature_alg = GaussLegendre(n = 5)))
    qprob = discretize(pde_system, qdisc)
    qmd = pinn_metadata(qprob)
    W = getp(qprob, qmd.blocks[1].w)(qprob)
    @test size(W) == (1, 25)
    @test sum(W) ≈ 1.0
    @test_throws ArgumentError discretize(
        pde_system, PhysicsInformedNN(chain, QuadratureTraining())
    )
end

@testset "initial parameters and eltype" begin
    ps = ComponentArrays.ComponentArray(Lux.initialparameters(Xoshiro(0), chain))
    idisc = PhysicsInformedNN(chain, GridTraining(0.1); init_params = collect(ps))
    iprob = discretize(pde_system, idisc)
    @test iprob.u0 == collect(ps)
    @test_throws ArgumentError PhysicsInformedNN(chain, GridTraining(0.1); init_params = [1.0]) |>
        d -> discretize(pde_system, d)
    fdisc = PhysicsInformedNN(chain, GridTraining(0.1); init_params = Float32.(collect(ps)))
    fprob = discretize(pde_system, fdisc)
    @test eltype(fprob.u0) == Float32
    @test fprob.f(fprob.u0, fprob.p) isa Float32
    for init in (Lux.initialparameters(Xoshiro(0), chain), ps, [ps])
        nprob = discretize(pde_system, PhysicsInformedNN(chain, GridTraining(0.1); init_params = init))
        @test nprob.u0 == collect(ps)
    end
end

optimization_reactant_loaded = try
    @eval using OptimizationReactant
    # the package can load without a functional XLA backend (e.g. 32-bit
    # platforms with no Reactant_jll client); probe the client the test needs
    OptimizationReactant.Reactant.to_rarray(Float64[1.0])
    true
catch
    false
end

if optimization_reactant_loaded
    @testset "AutoReactant backend" begin
        @test Base.get_extension(NeuralPDE, :NeuralPDEOptimizationReactantExt) !== nothing
        @test NeuralPDE.default_adtype() isa ADTypes.AutoReactant
        rprob = discretize(pde_system, disc)
        @test rprob.f.adtype isa ADTypes.AutoReactant
        res = solve(rprob, Adam(0.001); maxiters = 50)
        @test res.original_sol.objective < prob.f(θ, prob.p)
    end
end
