using NeuralPDE, SciMLBase
using Test

@testset "ODE RHS batch evaluation" begin
    using Lux, Optimisers, Random

    Random.seed!(100)
    chain = Chain(Dense(1, 4, tanh), Dense(4, 2))
    strategy = StochasticTraining(4)

    @test NNODE(chain, Adam(0.01); strategy).ode_batch_eval == false

    @testset "pointwise RHS" begin
        function pointwise_rhs(u::AbstractVector, p, t::Number)
            x, y = u
            return [y, -x]
        end

        pointwise_prob = ODEProblem(pointwise_rhs, [1.0, 0.0], (0.0, 1.0))
        pointwise_sol = solve(
            pointwise_prob,
            NNODE(chain, Adam(0.01); strategy, ode_batch_eval = false);
            verbose = false, maxiters = 1, saveat = [0.5]
        )
        @test length(pointwise_sol(0.5)) == 2
    end

    @testset "batched RHS opt-in" begin
        function batched_rhs(u, p, t::AbstractVector)
            return vcat(transpose(u[2, :]), transpose(-u[1, :]))
        end
        batched_rhs(u, p, t::Number) = error("pointwise evaluation used")

        batched_prob = ODEProblem(batched_rhs, [1.0, 0.0], (0.0, 1.0))
        batched_sol = solve(
            batched_prob,
            NNODE(chain, Adam(0.01); strategy, ode_batch_eval = true);
            verbose = false, maxiters = 1, saveat = [0.5]
        )
        @test length(batched_sol(0.5)) == 2
    end
end
