using DomainSets: Interval
using ModelingToolkit, NeuralPDE, SciMLBase
using Test

struct MinimalPINN <: NeuralPDE.AbstractPINN
    scale::Float64
end

function SciMLBase.symbolic_discretize(
        pde_system::PDESystem, discretization::MinimalPINN
    )
    return (; pde_system, scale = discretization.scale)
end

struct MinimalTrainingStrategy <: NeuralPDE.AbstractTrainingStrategy end

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
    @testset "AbstractPINN" begin
        @parameters x
        @variables u(..)
        @named pde_system = PDESystem(
            [u(x) ~ 0], [u(0) ~ 0], [x ∈ Interval(0, 1)], [x], [u(x)]
        )

        discretization = MinimalPINN(2.0)
        @test discretization isa NeuralPDE.AbstractPINN
        symbolic_problem = SciMLBase.symbolic_discretize(pde_system, discretization)
        @test symbolic_problem.pde_system === pde_system
        @test symbolic_problem.scale == 2.0
    end

    @testset "AbstractTrainingStrategy" begin
        strategy = MinimalTrainingStrategy()
        loss = NeuralPDE.get_loss_function(
            nothing, (data, θ) -> data .- θ, [1.0, 2.0], Float64, strategy;
            scale = 2.0
        )

        @test strategy isa NeuralPDE.AbstractTrainingStrategy
        @test loss([1.5, 2.0]) == 0.5
    end

    @testset "NeuralPDEAlgorithm" begin
        prob = SciMLBase.ODEProblem((u, p, t) -> zero(u), 1.0, (0.0, 1.0))
        alg = MinimalAlgorithm()
        sol = SciMLBase.solve(prob, alg)

        @test alg isa NeuralPDE.NeuralPDEAlgorithm
        @test sol.u == [1.0, 1.0]
        @test sol.retcode == SciMLBase.ReturnCode.Success
    end
end
