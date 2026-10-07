using NeuralPDE, SciMLBase
using Test

@testset "NNSDE callback keyword" begin
    using Lux, Optimisers, StableRNGs

    prob = SDEProblem((u, p, t) -> u, (u, p, t) -> u, 0.5, (0.0, 1.0))
    chain = Chain(Dense(4, 4, tanh), Dense(4, 1)) |> f64
    alg = NNSDE(chain, Adam(0.01); sub_batch = 2, numensemble = 4, rng = StableRNG(1))
    kw = (; dt = 0.5, maxiters = 1, verbose = false)

    sol = solve(prob, alg; kw..., callback = SciMLBase.CallbackSet())
    @test sol.original isa SciMLBase.OptimizationSolution

    cb = SciMLBase.DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
    @test_throws ErrorException solve(prob, alg; kw..., callback = cb)
end
