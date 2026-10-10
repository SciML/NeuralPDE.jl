using NeuralPDE, SciMLBase
using Test

@testset "BNNODE callback keyword" begin
    using AdvancedHMC, Lux, MCMCChains, Random

    Random.seed!(100)
    prob = ODEProblem((u, p, t) -> cos(2 * π * t), 0.0, (0.0, 1.0))
    alg = BNNODE(Chain(Dense(1, 4, tanh), Dense(4, 1)); draw_samples = 10)

    sol = solve(prob, alg; callback = SciMLBase.CallbackSet())
    @test sol isa BPINNsolution

    cb = SciMLBase.DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
    @test_throws ErrorException solve(prob, alg; callback = cb)
end
