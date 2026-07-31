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

using NeuralPDE
using Test

@testset "Example multiple parameters Сhain du = p1 * cos(p2 * t)" begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators, Random
    equation = (u, p, t) -> p[1] * cos(p[2] * t) #+ p[3]
    tspan = (0.0, 1.0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)

    input_branch_size = 2
    chain = Chain(
        Dense(input_branch_size + 1 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast), Dense(10 => 1)
    )

    x = rand(Float32, 3, 1000, 10)
    θ, st = Lux.setup(Random.default_rng(), chain)
    c = chain(x, θ, st)[1]

    bounds = [(1.0, pi), (1.0, 2.0)] #, (2.0, 3.0)]
    number_of_parameters = 200
    strategy = StochasticTraining(200)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(chain, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 3000)

    ground_solution = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t) #+ p[3] * t

    function ground_solution_f(p, t)
        reduce(
            hcat,
            [
                [ground_solution(u0, p[:, i, j], t[1, i, j]) for j in axes(t, 3)]
                    for i in axes(p, 2)
            ]
        )'
    end
    (p, t) = get_trainset(chain, bounds, 20, tspan, 0.1)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp(p, t)[1, :, :]
    @test ground_solution_ ≈ predict rtol = 0.08

    p, t = get_trainset(chain, bounds, 50, tspan, 0.025)
    ground_solution_ = ground_solution_f(p, t)
    predict_sol = sol.interp(p, t)[1, :, :]
    @test ground_solution_ ≈ predict_sol rtol = 0.08
end

#multiple parameters DeepOnet
