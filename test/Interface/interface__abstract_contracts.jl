using DomainSets: Interval
using ForwardDiff: derivative
using Integrals: CubatureJLh
using ModelingToolkit, NeuralPDE, SciMLBase, Lux, SymbolicIndexingInterface
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
end
