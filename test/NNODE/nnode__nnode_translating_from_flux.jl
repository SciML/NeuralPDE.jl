using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "NNODE: Translating from Flux" begin
    using OrdinaryDiffEq, Random, Lux, Optimisers
    import Flux

    Random.seed!(100)

    linear = (u, p, t) -> cospi(2t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sinpi(2t)
    tspan = (0.0f0, 1.0f0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(
        ODEFunction(linear; analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0)
    )

    u_analytical(x) = (1 / (2pi)) .* sinpi.(2x)
    fluxchain = Flux.Chain(Flux.Dense(1, 5, Flux.σ), Flux.Dense(5, 1))
    opt = Adam(0.1)

    alg = NNODE(fluxchain, opt)
    @test alg.chain isa Lux.AbstractLuxLayer
    # `NNODE` must be recognized as a SciML algorithm, otherwise `solve` silently
    # falls back to the default ODE algorithm (see the `DiffEqBase.extract_alg`
    # path exercised whenever OrdinaryDiffEq is loaded alongside NeuralPDE).
    @test alg isa SciMLBase.AbstractODEAlgorithm
    sol = solve(prob, alg; verbose = false, abstol = 1.0e-10, maxiters = 200)
    @test sol.alg isa NNODE
    @test sol.errors[:l2] < 0.5
end
