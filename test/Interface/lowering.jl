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

@testset "general arguments lower with nn_vcat and FD composition" begin
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    # Exact network identity: apply(X, θ) = X, so Dx(u(2x)) = 2 at every point.
    chain = Chain(Dense(1, 1))
    init = [1.0, 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named scaled = PDESystem(
        [Dx(u(2x)) ~ 2.0], [u(0.0) ~ 0.0], domains, [x], [u(x)]
    )
    sdisc = PhysicsInformedNN(chain, GridTraining(0.25); init_params = init, rng = Xoshiro(3))
    sprob = discretize(scaled, sdisc)
    smd = pinn_metadata(sprob)
    residual = getu(sprob, smd.blocks[1].residual)(sprob)
    @test residual ≈ zeros(size(residual)) atol = 1.0e-8

    apply = getdefault(only(smd.networks).NN)
    X = getp(sprob, smd.blocks[1].xs)(sprob)
    ε = eps(Float64)^(1 / 3)
    # Manual FD of the composition x ↦ apply(2x): (f(2(x+ε)) - f(2(x-ε))) / (2ε).
    manual = (apply(2 .* (X .+ ε), init) .- apply(2 .* (X .- ε), init)) ./ (2ε)
    @test residual .+ 2 ≈ manual atol = 1.0e-8
    @test manual ≈ fill(2.0, size(manual)) atol = 1.0e-8

    @named shifted = PDESystem(
        [u(x + 1) - u(x) ~ 1.0], [u(0.0) ~ 0.0], domains, [x], [u(x)]
    )
    shdisc = PhysicsInformedNN(chain, GridTraining(0.25); init_params = init, rng = Xoshiro(4))
    shprob = discretize(shifted, shdisc)
    shmd = pinn_metadata(shprob)
    # Identity network: u(x + 1) - u(x) = 1.
    @test getu(shprob, shmd.blocks[1].residual)(shprob) ≈
        zeros(1, size(getp(shprob, shmd.blocks[1].xs)(shprob), 2)) atol = 1.0e-12

    # Two-input network; both inputs enter the analytic expectation so an incorrect
    # second row (e.g. `x` instead of `1 - x`) cannot cancel.
    @parameters t
    chain2 = Chain(Dense(2, 1))
    init2 = [3.0, 5.0, 0.0]  # weights [3, 5], bias 0 → output = 3a + 5b
    @named mixed2 = PDESystem(
        [u(t, 1 - x) ~ 3t + 5(1 - x)], [u(0.0, 0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]
    )
    mdisc = PhysicsInformedNN(chain2, GridTraining(0.25); init_params = init2, rng = Xoshiro(8))
    mprob = discretize(mixed2, mdisc)
    mmd = pinn_metadata(mprob)
    @test occursin("nn_vcat", string(mmd.blocks[1].residual))
    @test getu(mprob, mmd.blocks[1].residual)(mprob) ≈
        zeros(size(getu(mprob, mmd.blocks[1].residual)(mprob))) atol = 1.0e-12

    # Literal + general composition: Dx(u(0, 2x)) must keep the literal fixed.
    # Declared signature u(x, y) so the literal occupies the differentiated slot;
    # exact network u(a,b)=3a+5b ⇒ u(0,2x)=10x ⇒ Dx = 10 (not 13 from shifting 0).
    @parameters y
    chain_xy = Chain(Dense(2, 1))
    init_xy = [3.0, 5.0, 0.0]
    @named litgen = PDESystem(
        [Dx(u(0.0, 2x)) ~ 10.0], [u(0.0, 0.0) ~ 0.0],
        [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)], [x, y], [u(x, y)]
    )
    lgdisc = PhysicsInformedNN(chain_xy, GridTraining(0.25); init_params = init_xy, rng = Xoshiro(9))
    lgprob = discretize(litgen, lgdisc)
    lgmd = pinn_metadata(lgprob)
    lg_residual = getu(lgprob, lgmd.blocks[1].residual)(lgprob)
    @test lg_residual ≈ zeros(size(lg_residual)) atol = 1.0e-7
    # Manual FD of the composition x ↦ apply([0; 2x]): literal row stays 0.
    apply_xy = getdefault(only(lgmd.networks).NN)
    Xl = getp(lgprob, lgmd.blocks[1].xs)(lgprob)  # free ivs of Dx(u(0,2x)): only x
    ε1 = eps(Float64)^(1 / 3)
    Xp = vcat(zeros(1, size(Xl, 2)), 2 .* (Xl .+ ε1))
    Xm = vcat(zeros(1, size(Xl, 2)), 2 .* (Xl .- ε1))
    manual_litgen = (apply_xy(Xp, init_xy) .- apply_xy(Xm, init_xy)) ./ (2ε1)
    @test lg_residual .+ 10 ≈ manual_litgen atol = 1.0e-7
    @test manual_litgen ≈ fill(10.0, size(manual_litgen)) atol = 1.0e-7

    # Plain calls nested in a general argument keep their literals fixed under an outer
    # FD; a derivative inside the argument still uses the plain boundary convention.
    # Expected values are hand algebra of u(a,b)=3a+5b, e.g. u(u(0,y),2x) = 15y + 10x.
    xy_domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    xy_residual = function (eq)
        sys = PDESystem(
            [eq], [u(0.0, 0.0) ~ 0.0], xy_domains, [x, y], [u(x, y)]; name = :nested
        )
        pdisc = PhysicsInformedNN(
            chain_xy, GridTraining(0.25); init_params = init_xy, rng = Xoshiro(10)
        )
        p = discretize(sys, pdisc)
        return getu(p, pinn_metadata(p).blocks[1].residual)(p)
    end
    for eq in [
            Dx(u(u(0.0, y), 2x)) ~ 10.0,
            Dx(u(u(0.0, x), 2x)) ~ 25.0,
            u(Dx(u(0.0, y)), 2x) ~ 9 + 10x,
            Dx(u(Dx(u(0.0, y)), 2x)) ~ 10.0,
        ]
        r = xy_residual(eq)
        @test r ≈ zeros(size(r)) atol = 1.0e-7
    end
    # Differentiating along a variable that only occurs in fixed literal slots would be
    # a θ-independent residual.
    for eq in [Dx(u(0.0, 2y)) ~ 0.0, Dx(u(u(0.0, y), 2y)) ~ 0.0]
        @test_throws ArgumentError xy_residual(eq)
    end

    @named reflected = PDESystem(
        [u(1 - x) ~ u(x)], [u(0.0) ~ 0.0], domains, [x], [u(x)]
    )
    rdisc = PhysicsInformedNN(chain, GridTraining(0.25); init_params = init, rng = Xoshiro(4))
    rprob = discretize(reflected, rdisc)
    rmd = pinn_metadata(rprob)
    Xr = getp(rprob, rmd.blocks[1].xs)(rprob)
    # Identity network: u(1 - x) - u(x) = (1 - x) - x = 1 - 2x.
    @test getu(rprob, rmd.blocks[1].residual)(rprob) ≈ 1 .- 2 .* Xr

    @named nested = PDESystem(
        [u(2 * (1 - x)) ~ 2 * (1 - x)], [u(0.0) ~ 0.0], domains, [x], [u(x)]
    )
    ndisc = PhysicsInformedNN(chain, GridTraining(0.25); init_params = init, rng = Xoshiro(5))
    nprob = discretize(nested, ndisc)
    nmd = pinn_metadata(nprob)
    Xn = getp(nprob, nmd.blocks[1].xs)(nprob)
    @test getu(nprob, nmd.blocks[1].residual)(nprob) ≈ zeros(size(Xn)) atol = 1.0e-12

    # Plain arguments keep the affine `P * X + c` shortcut (no nn_vcat).
    @named plain = PDESystem(
        [Dx(u(x)) ~ 1.0], [u(0.0) ~ 0.0], domains, [x], [u(x)]
    )
    pdisc = PhysicsInformedNN(chain, GridTraining(0.25); init_params = init, rng = Xoshiro(6))
    pprob = discretize(plain, pdisc)
    @test !occursin("nn_vcat", string(pinn_metadata(pprob).blocks[1].residual))
end
