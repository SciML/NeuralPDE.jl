using NeuralPDE, SciMLBase
using Test

@testset "PINOODE callback keyword" begin
    using Lux, Optimisers, StableRNGs

    prob = ODEProblem((u, p, t) -> cos(p * t), 1.0, (0.0, 1.0))
    chain = Chain(Dense(2 => 8, tanh), Dense(8 => 1))
    alg = PINOODE(
        chain, Adam(0.01), [(pi, 2pi)], 10; strategy = StochasticTraining(10),
        rng = StableRNG(1)
    )

    sol = solve(prob, alg; verbose = false, maxiters = 2, callback = SciMLBase.CallbackSet())
    @test sol.retcode == ReturnCode.Success

    cb = SciMLBase.DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
    @test_throws ErrorException solve(prob, alg; verbose = false, maxiters = 2, callback = cb)
end
