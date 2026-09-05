using NeuralPDE, SciMLBase
using Test

@testset "ODE RHS batch evaluation" begin
    using Lux, Optimisers, Random

    Random.seed!(100)
    chain = Chain(Dense(1, 4, tanh), Dense(4, 2))
    strategy = StochasticTraining(4)

    @testset "pointwise RHS opt-out" begin
        function pointwise_rhs(u::AbstractVector, p, t::Number)
            return [u[2], -u[1]]
        end

        prob = ODEProblem(pointwise_rhs, [1.0, 0.0], (0.0, 1.0))
        alg = NNODE(chain, Adam(0.01); strategy, ode_batch_eval = false)
        sol = solve(
            prob, alg; verbose = false, maxiters = 1, saveat = [0.5],
            tstops = [0.25, 0.75]
        )
        @test length(sol(0.5)) == 2
    end

    @testset "batched RHS default" begin
        function batched_rhs(u::AbstractMatrix, p, t::AbstractVector)
            return vcat(u[2:2, :], -u[1:1, :])
        end
        batched_rhs(u::AbstractVector, p, t::Number) = error("pointwise evaluation used")

        prob = ODEProblem(batched_rhs, [1.0, 0.0], (0.0, 1.0))
        alg = NNODE(chain, Adam(0.01); strategy)
        @test alg.ode_batch_eval

        sol = solve(
            prob, alg; verbose = false, maxiters = 1, saveat = [0.5],
            tstops = [0.25, 0.75]
        )
        @test length(sol(0.5)) == 2
    end
end
