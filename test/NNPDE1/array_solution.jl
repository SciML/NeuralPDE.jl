using DomainSets: Interval
using Lux: Dense
using ModelingToolkit, NeuralPDE, Test
using OptimizationOptimisers: Adam

@testset "solution indexing for array-valued dependent variables" begin
    @parameters x
    @variables u(..)[1:2]
    a, b = collect(u(x))
    @named system = PDESystem(
        [a ~ x, b ~ 2 - x], [u(0)[1] ~ 0, u(0)[2] ~ 2],
        [x ∈ Interval(0.0, 1.0)], [x], [u(x)]
    )
    disc = PhysicsInformedNN(
        Dense(1 => 2), GridTraining(0.25);
        init_params = [1.0, -1.0, 0.0, 2.0], eval_points = 5
    )
    sol = solve(discretize(system, disc), Adam(0.01); maxiters = 1)
    for (dv, f) in ((a, identity), (b, x -> 2 - x))
        @test sol[dv] == f.(sol[x])
        @test sol[dv, 2] == f(0.25)
        @test sol(0.375; dv) == f(0.375)
    end
end
