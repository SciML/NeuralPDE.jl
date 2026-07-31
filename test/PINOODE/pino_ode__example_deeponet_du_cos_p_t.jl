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

@testset "Example DeepONet du = cos(p * t)" begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators, Random
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0, 1.0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)
    deeponet = NeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)
        ),
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)
        )
    )
    bounds = [(pi, 2pi)]
    number_of_parameters = 50
    strategy = StochasticTraining(40)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 3000)
    ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    p, t = get_trainset(deeponet, bounds, 50, tspan, 0.025)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08
    p, t = get_trainset(deeponet, bounds, 100, tspan, 0.01)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08

    # NeuralOperators 0.6+ requires 2D trunk input
    p, t = sol.p, rand(1, 20)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol(t)
    @test ground_solution ≈ predict_sol rtol = 0.08
end
