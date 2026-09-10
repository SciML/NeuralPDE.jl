module PINOODETestSetup
    using Lux, NeuralOperators

    function get_trainset(chain::DeepONet, bounds, number_of_parameters, tspan, dt)
        p_ = [range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]
        p = vcat([collect(reshape(p_i, 1, size(p_i, 1))) for p_i in p_]...)
        t_ = collect(tspan[1]:dt:tspan[2])
        # NeuralOperators 0.6+ requires 2D trunk input
        t = reshape(t_, 1, size(t_, 1))
        return (p, t)
    end

    function get_trainset(chain::Lux.Chain, bounds, number_of_parameters, tspan, dt)
        tspan_ = tspan[1]:dt:tspan[2]
        pspan = [
            range(start = b[1], length = number_of_parameters, stop = b[2])
                for b in bounds
        ]
        x_ = hcat(
            vec(
                map(
                    points -> collect(points), Iterators.product([pspan..., tspan_]...)
                )
            )...
        )
        x = reshape(x_, size(bounds, 1) + 1, prod(size.(pspan, 1)), size(tspan_, 1))
        p, t = x[1:(end - 1), :, :], x[[end], :, :]
        return (p, t)
    end
    export get_trainset
end
#Test Chain

using .PINOODETestSetup

using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "Example Chain du = cos(p * t)" begin
    using ModelingToolkit, NeuralPDE, SciMLBase, Lux, OptimizationOptimisers, NeuralOperators, Random,
        StableRNGs
    using SciMLBase: SciMLBase, PDETimeSeriesSolution
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0, 1.0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)
    chain = Chain(
        Dense(2 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 1)
    )
    bounds = [(pi, 2pi)]
    number_of_parameters = 300
    strategy = StochasticTraining(300)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(
        chain, opt, bounds, number_of_parameters;
        strategy, rng = StableRNG(1)
    )
    legacy_alg = PINOODE(
        chain, opt, bounds, number_of_parameters,
        nothing, strategy, nothing, (;)
    )
    @test legacy_alg isa PINOODE
    sol = solve(prob, alg, verbose = false, maxiters = 3000)

    # Solution type contract: a `PINOODE` solve is a PDE solve where the
    # ODE parameters are extra PDE dimensions, so the result is a
    # `PDETimeSeriesSolution` tagged with `PINOODEMetadata` — *not* an
    # `ODESolution`. `sol.prob` is the user's original ODEProblem, not a
    # fake one with the parameter sample tensor stuffed in.
    @test sol isa PDETimeSeriesSolution
    @test sol isa SciMLBase.AbstractPDETimeSeriesSolution
    @test !(sol isa SciMLBase.AbstractODESolution)
    @test sol.prob === prob
    @test sol.alg === alg
    @test sol.retcode == SciMLBase.ReturnCode.Success

    ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    p, t = get_trainset(chain, bounds, 50, tspan, 0.025)
    ground_solution = ground_analytic.(u0, p, t)
    predict_sol = sol.interp(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08
    p, t = get_trainset(chain, bounds, 100, tspan, 0.01)
    ground_solution = ground_analytic.(u0, p, t)
    predict_sol = sol.interp(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08

    p = reshape(collect(range(pi, 2pi; length = 100)), 1, :)
    t = fill(1.0, size(p))
    ground_solution = ground_analytic.(u0, p, t)
    predict_sol = sol(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08
    @test sol(1.0) == sol.interp(sol.p, fill(1.0, size(sol.p)))

    t = reshape(collect(range(0.0, 1.0; length = length(p))), size(p))
    ground_solution = ground_analytic.(u0, p, t)
    predict_sol = sol.interp(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08

    # Explicit PDE-style (p, t) call form.
    @test sol(p, t) == predict_sol
    training_t = reshape(collect(range(0.0, 1.0; length = length(sol.p))), size(sol.p))
    @test sol(training_t) == sol(sol.p, training_t)

    function short_solve(ambient_seed, algorithm_seed)
        Random.seed!(ambient_seed)
        short_chain = Chain(Dense(2 => 3, tanh), Dense(3 => 1))
        short_alg = PINOODE(
            short_chain, opt, bounds, 4;
            strategy = StochasticTraining(4), rng = StableRNG(algorithm_seed)
        )
        return solve(prob, short_alg, verbose = false, maxiters = 1)
    end

    first_sol = short_solve(1, 7)
    second_sol = short_solve(2, 7)
    different_sol = short_solve(1, 8)
    @test first_sol.p == second_sol.p
    @test first_sol.u == second_sol.u
    @test first_sol.p != different_sol.p
end

#Test DeepONet
