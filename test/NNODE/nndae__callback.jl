using NeuralPDE, SciMLBase
using Test

@testset "NNDAE callback keyword" begin
    using Lux, Optimisers

    example = (du, u, p, t) -> [cos(2pi * t) - du[1], u[2] + cos(2pi * t) - du[2]]
    prob = DAEProblem(
        example, [0.0, 0.0], [1.0, -1.0], (0.0, 1.0); differential_vars = [true, false]
    )
    alg = NNDAE(Chain(Dense(1, 8, cos), Dense(8, 2)), Adam(0.01); autodiff = false)
    kw = (; verbose = false, dt = 1 / 10, maxiters = 2)

    sol = solve(prob, alg; kw..., callback = SciMLBase.CallbackSet())
    @test sol.retcode == ReturnCode.Success

    cb = SciMLBase.DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
    @test_throws ErrorException solve(prob, alg; kw..., callback = cb)
end
