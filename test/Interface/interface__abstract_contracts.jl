using DomainSets: Interval
using ForwardDiff: derivative
using Integrals: CubatureJLh
using ModelingToolkit, NeuralPDE, SciMLBase, Lux, SymbolicIndexingInterface
using Optimisers: Adam
using Test

struct MinimalTrainingStrategy <: NeuralPDE.AbstractTrainingStrategy end

struct FixedPointsTraining <: NeuralPDE.AbstractTrainingStrategy end

NeuralPDE.collocation_count(::FixedPointsTraining, kind, ivpos, bounds, pinned) =
    isempty(ivpos) ? 1 : 3
function NeuralPDE.sample_points(::FixedPointsTraining, block::NeuralPDE.ResidualBlock, rng)
    lb, ub = block.bounds
    return lb .+ (ub .- lb) .* [0.25 0.5 0.75], nothing
end

function NeuralPDE.get_loss_function(
        init_params, loss_function, training_data, eltypeθ,
        ::MinimalTrainingStrategy; scale = one(eltypeθ)
    )
    return θ -> scale * sum(abs2, loss_function(training_data, θ))
end

struct MinimalAlgorithm <: NeuralPDE.NeuralPDEAlgorithm end

# Stands in for a verbosity object such as `DiffEqBase.DEVerbosity`, which the solvers
# cannot interpret and must not use in boolean context.
struct OpaqueVerbosity end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractODEProblem, ::MinimalAlgorithm; kwargs...
    )
    t = collect(prob.tspan)
    u = [prob.u0, prob.u0]
    return SciMLBase.build_solution(
        prob, MinimalAlgorithm(), t, u;
        dense = false, retcode = SciMLBase.ReturnCode.Success
    )
end

@testset "Developer interface contracts" begin
    @testset "PDE collocation strategy contract" begin
        @test Docs.doc(NeuralPDE.collocation_count) !== nothing
        @test Docs.doc(NeuralPDE.sample_points) !== nothing
        @parameters x
        @variables u(..)
        @named pde_system = PDESystem(
            [Differential(x)(u(x)) ~ 1], [u(0) ~ 0], [x ∈ Interval(0, 1)], [x], [u(x)]
        )
        strategy = FixedPointsTraining()
        disc = PhysicsInformedNN(Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1)), strategy)
        prob = discretize(pde_system, disc)
        md = pinn_metadata(prob)
        @test md.blocks[1].npoints == 3
        @test SymbolicIndexingInterface.getp(prob, md.blocks[1].xs)(prob) == [0.25 0.5 0.75]
        @test prob.f(prob.u0, prob.p) isa Float64
    end

    @testset "AbstractTrainingStrategy" begin
        @test Docs.doc(NeuralPDE.AbstractTrainingStrategy) !== nothing
        @test Docs.doc(NeuralPDE.get_loss_function) !== nothing
        strategy = MinimalTrainingStrategy()
        loss = NeuralPDE.get_loss_function(
            nothing, (data, θ) -> data .- θ, [1.0, 2.0], Float64, strategy;
            scale = 2.0
        )

        @test strategy isa NeuralPDE.AbstractTrainingStrategy
        @test loss([1.5, 2.0]) == 0.5
    end

    @testset "QuadratureTraining skips empty prototype batches" begin
        calls = Ref(0)
        residuals = function (x, θ)
            isempty(x) && error("loss function cannot evaluate an empty batch")
            calls[] += 1
            return θ[1] .* x
        end
        strategy = QuadratureTraining(
            quadrature_alg = CubatureJLh(), reltol = 1.0e-8, abstol = 1.0e-8,
            maxiters = 10_000, batch = 16
        )
        loss = get_loss_function(
            [2.0], residuals, [0.0], [1.0], Float64, strategy
        )

        @test only(loss([2.0])) ≈ 4 / 3
        @test derivative(t -> only(loss([t])), 2.0) ≈ 4 / 3
        @test calls[] > 0
    end

    @testset "NeuralPDEAlgorithm" begin
        @test Docs.doc(NeuralPDE.NeuralPDEAlgorithm) !== nothing
        prob = SciMLBase.ODEProblem((u, p, t) -> zero(u), 1.0, (0.0, 1.0))
        alg = MinimalAlgorithm()
        sol = SciMLBase.solve(prob, alg)

        @test alg isa NeuralPDE.NeuralPDEAlgorithm
        @test sol.u == [1.0, 1.0]
        @test sol.retcode == SciMLBase.ReturnCode.Success
    end

    # `DiffEqBase.extract_alg` only picks up the algorithm passed to `solve` when it
    # is a `SciMLBase.AbstractSciMLAlgorithm`. Anything else is treated as "no
    # algorithm given" and is silently replaced by the default solver of the loaded
    # solver package, so the NeuralPDE solver would never run.
    @testset "Solver algorithms subtype the SciMLBase algorithm hierarchy" begin
        for T in (NeuralPDE.NNODE, NeuralPDE.PINOODE, NeuralPDE.BNNODE)
            @test T <: SciMLBase.AbstractODEAlgorithm
        end

        @test NeuralPDE.NNDAE <: SciMLBase.AbstractDAEAlgorithm

        for T in (NeuralPDE.NNSDE, NeuralPDE.SDEPINN)
            @test T <: SciMLBase.AbstractSDEAlgorithm
        end
    end

    # On Julia 1.12 and later `DiffEqBase.merge_problem_kwargs` rewrites `callback`
    # into a type-erased `CallbackSet{Vector{Any}, Vector{Any}}` before dispatching to
    # `__solve`, and it does so even for a `solve` call that passed no callback, so a
    # solver that does not accept the keyword fails with a `MethodError`. The keyword
    # is passed explicitly here so the contract is checked on every Julia version.
    @testset "Solvers accept the callback keyword they cannot honour" begin
        erased = SciMLBase.CallbackSet(Any[], Any[])
        stepping = SciMLBase.DiscreteCallback(
            (u, t, integrator) -> false, integrator -> nothing
        )
        alg = MinimalAlgorithm()

        # Nothing to report: no callback, or the empty set synthesised by the erasure.
        @test_logs NeuralPDE.warn_unsupported_callback(alg, nothing)
        @test_logs NeuralPDE.warn_unsupported_callback(alg, erased)
        @test_logs NeuralPDE.warn_unsupported_callback(alg, SciMLBase.CallbackSet())

        # A callback the caller supplied is dropped, but not silently. It arrives
        # wrapped in a set when something merged it, and bare when nothing did.
        warning = (:warn, r"`callback` keyword is ignored")
        @test_logs warning NeuralPDE.warn_unsupported_callback(
            alg, SciMLBase.CallbackSet(stepping)
        )
        @test_logs warning NeuralPDE.warn_unsupported_callback(alg, stepping)

        prob = SciMLBase.ODEProblem((u, p, t) -> zero(u), 0.0, (0.0, 1.0))
        nnode = NeuralPDE.NNODE(Chain(Dense(1, 3, tanh), Dense(3, 1)), Adam(0.01))
        for cb in (nothing, erased)
            sol = SciMLBase.solve(
                prob, nnode; dt = 0.5, maxiters = 2, verbose = false, callback = cb
            )
            @test sol.retcode isa SciMLBase.ReturnCode.T
        end
    end

    # `DiffEqBase.solve` defaults `verbose` to a `DEVerbosity` verbosity object and
    # forwards it on every Julia version, so a `solve` call that passes no `verbose` of
    # its own reaches `__solve` with a value that cannot be used in boolean context.
    @testset "Solvers reduce a verbosity object to their quiet default" begin
        @test NeuralPDE.verbose_flag(true)
        @test !NeuralPDE.verbose_flag(false)
        # Only a `Bool` says anything about the one message `verbose` controls here.
        @test !NeuralPDE.verbose_flag(OpaqueVerbosity())
        @test !NeuralPDE.verbose_flag(nothing)

        # Omitting `verbose` is what puts the real `DEVerbosity` object in its place.
        prob = SciMLBase.ODEProblem((u, p, t) -> zero(u), 0.0, (0.0, 1.0))
        nnode = NeuralPDE.NNODE(Chain(Dense(1, 3, tanh), Dense(3, 1)), Adam(0.01))
        sol = SciMLBase.solve(prob, nnode; dt = 0.5, maxiters = 2)
        @test sol.retcode isa SciMLBase.ReturnCode.T
    end
end
