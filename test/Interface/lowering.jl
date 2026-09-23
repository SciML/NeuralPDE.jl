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
    default = Sys.WORD_SIZE == 32 ? AutoZygote : AutoEnzyme
    @test prob.f.adtype isa default
    dgm_disc = DeepGalerkin(
        2, 1, 4, 1, tanh, tanh, identity, GridTraining(0.5); rng = Xoshiro(2)
    )
    @test discretize(pde_system, dgm_disc).f.adtype isa AutoZygote
    @test discretize(pde_system, dgm_disc; adtype = AutoEnzyme()).f.adtype isa AutoEnzyme
    nested_dgm = PhysicsInformedNN(
        Chain(NeuralPDE.DGM(2, 1, 4, 1, tanh, tanh, identity)), GridTraining(0.5);
        rng = Xoshiro(2)
    )
    @test discretize(pde_system, nested_dgm).f.adtype isa AutoZygote
    loss_disc = PhysicsInformedNN(
        chain, GridTraining(0.5); rng = Xoshiro(4), additional_loss = function (phi, θ, p)
            out = phi.u([0.5 0.25; 0.5 0.25], θ.u)
            v = similar(out, 2)
            v[1] = out[1]
            v[2] = out[2]
            return sum(abs2, v)
        end
    )
    loss_prob = discretize(pde_system, loss_disc)
    @test loss_prob.f.adtype isa default
    Sys.WORD_SIZE == 64 &&
        @test solve(loss_prob, Adam(0.001); maxiters = 2).original_sol.objective isa Real
    data = [0.3 0.1]
    capture_disc = PhysicsInformedNN(
        chain, GridTraining(0.5); rng = Xoshiro(5), additional_loss = function (phi, θ, p)
            return sum(abs2, phi.u([0.5 0.25; 0.5 0.25], θ.u) .- data)
        end
    )
    @test discretize(pde_system, capture_disc).f.adtype isa AutoZygote
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
