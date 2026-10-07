using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "callback keyword" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers

    Random.seed!(100)

    linear = (u, p, t) -> @. -u + t
    prob = ODEProblem(ODEFunction(linear), [1.0f0], (0.0f0, 1.0f0))
    alg = NNODE(Chain(Dense(1, 8, σ), Dense(8, 1)), Adam(0.01))

    # `solve` merges an empty `CallbackSet` into the `__solve` keyword
    # arguments (always on Julia >= 1.12), so `NNODE` must accept it.
    sol = solve(
        prob, alg; verbose = false, maxiters = 10,
        callback = SciMLBase.CallbackSet()
    )
    @test sol.retcode == ReturnCode.Success

    # Real ODE callbacks cannot be honored by a non-integrator solver.
    cb = SciMLBase.DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
    @test_throws ErrorException solve(
        prob, alg; verbose = false, maxiters = 10, callback = cb
    )
end
