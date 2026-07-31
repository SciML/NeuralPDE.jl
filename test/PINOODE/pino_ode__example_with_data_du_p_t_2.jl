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

@testset "Example with data du = p*t^2" begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators, Random
    equation = (u, p, t) -> p * t^2
    tspan = (0.0, 1.0)
    u0 = 0.0
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
    bounds = [(0.0, 10.0)]
    number_of_parameters = 60
    dt = (tspan[2] - tspan[1]) / 40
    strategy = StochasticTraining(60)
    opt = OptimizationOptimisers.Adam(0.01)

    #generate data
    ground_analytic = (u0, p, t) -> u0 + p * t^3 / 3
    p, t = get_trainset(deeponet, bounds, number_of_parameters, tspan, dt)
    sol = ground_analytic.(u0, p, vec(t))
    function additional_loss_(phi, θ)
        u = phi((p, t), θ)
        norm = prod(size(u))
        sum(abs2, u .- sol) / norm
    end

    alg = PINOODE(
        deeponet, opt, bounds, number_of_parameters; strategy = strategy,
        additional_loss = additional_loss_
    )
    sol = solve(prob, alg, verbose = false, maxiters = 3000)

    p, t = get_trainset(deeponet, bounds, number_of_parameters, tspan, dt)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp(p, t)
    @test ground_solution ≈ predict_sol rtol = 0.08
end

#multiple parameters Сhain
