using NeuralPDE, SciMLBase
using Test

@testset "Explicit NNSDE RNG" begin
    using Lux, Optimisers, Random, StableRNGs

    f(u, p, t) = u
    g(u, p, t) = u
    prob = SDEProblem(f, g, 0.5, (0.0, 1.0))
    chain = Chain(Dense(4, 4, tanh), Dense(4, 1)) |> f64
    opt = Adam(0.01)

    legacy_alg = NNSDE(
        chain, opt, nothing, nothing, false, true, 1, false, false, false,
        [], 1, 4, nothing, (;)
    )
    @test legacy_alg isa NNSDE

    function solve_once(ambient_seed, algorithm_seed)
        Random.seed!(ambient_seed)
        alg = NNSDE(
            chain, opt; sub_batch = 2, numensemble = 4, rng = StableRNG(algorithm_seed)
        )
        return solve(prob, alg; dt = 0.5, maxiters = 1, verbose = false)
    end

    sol_1 = solve_once(1, 100)
    sol_2 = solve_once(2, 100)

    @test sol_1.training_sets == sol_2.training_sets
    @test sol_1.ensemble_inputs == sol_2.ensemble_inputs
    @test sol_1.original.u == sol_2.original.u
end
