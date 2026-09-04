using NeuralPDE, SciMLBase
using ChainRulesCore: ignore_derivatives
using Test

@testset "ODE RHS batch evaluation" begin
    using Lux, Optimisers, Random

    Random.seed!(100)
    batch_widths = Int[]
    record_batch(x) = begin
        ignore_derivatives() do
            x isa AbstractMatrix && push!(batch_widths, size(x, 2))
        end
        return x
    end
    chain = Chain(WrappedFunction(record_batch), Dense(1, 4, tanh), Dense(4, 2))
    strategy = StochasticTraining(4)

    @test NNODE(chain, Adam(0.01); strategy).ode_batch_eval

    @testset "ordinary pointwise RHS" begin
        function pointwise_rhs(u::AbstractVector, p, t::Number)
            x, y = u
            return [y, -x]
        end

        prob = ODEProblem(pointwise_rhs, [1.0, 0.0], (0.0, 1.0))
        sol = solve(
            prob, NNODE(chain, Adam(0.01); strategy);
            verbose = false, maxiters = 1, saveat = [0.5]
        )
        @test length(sol(0.5)) == 2
    end
    @test strategy.points in batch_widths

    @testset "internal batched RHS" begin
        batched_rhs(u::AbstractVector, p, t::Number) = error("pointwise evaluation used")
        batched_rhs(u::AbstractMatrix, p, t::AbstractVector) = vcat(u[2:2, :], -u[1:1, :])

        prob = ODEProblem(NeuralPDE.BatchedRHS(batched_rhs), [1.0, 0.0], (0.0, 1.0))
        sol = solve(
            prob, NNODE(chain, Adam(0.01); strategy);
            verbose = false, maxiters = 1, saveat = [0.5]
        )
        @test length(sol(0.5)) == 2
    end
end
