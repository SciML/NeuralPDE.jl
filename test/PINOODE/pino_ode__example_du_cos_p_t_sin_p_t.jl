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

@testset "Example du = [cos(p * t), sin(p * t)]" begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators, Random
    equation = (u, p, t) -> [cos(p * t), sin(p * t)]
    tspan = (0.0, 1.0)
    u0 = [1.0, 0.0]
    prob = ODEProblem(equation, u0, tspan)
    input_branch_size = 1
    chain = Chain(
        Dense(input_branch_size + 1 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast), Dense(10 => 2)
    )

    bounds = [(pi, 2pi)]
    number_of_parameters = 100
    strategy = StochasticTraining(100)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(chain, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 3000)

    ground_solution = (u0, p, t) -> [1 + sin(p * t) / p, 1 / p - cos(p * t) / p]
    function ground_solution_f(p, t)
        ans_1 = reduce(
            hcat,
            [
                reduce(
                        vcat,
                        [ground_solution(u0, p[1, i, 1], t[1, 1, j])[1] for i in axes(p, 2)]
                    )
                    for j in axes(t, 3)
            ]
        )
        ans_2 = reduce(
            hcat,
            [
                reduce(
                        vcat,
                        [ground_solution(u0, p[1, i, 1], t[1, 1, j])[2] for i in axes(p, 2)]
                    )
                    for j in axes(t, 3)
            ]
        )

        ans_1 = reshape(ans_1, 1, size(ans_1)...)
        ans_2 = reshape(ans_2, 1, size(ans_2)...)
        vcat(ans_1, ans_2)
    end
    p, t = get_trainset(chain, bounds, 50, tspan, 0.025)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp(p, t)
    @test ground_solution_[1, :, :] ≈ predict[1, :, :] rtol = 0.08
    @test ground_solution_[2, :, :] ≈ predict[2, :, :] rtol = 0.08
    @test ground_solution_ ≈ predict rtol = 0.08

    p, t = get_trainset(chain, bounds, 300, tspan, 0.01)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp(p, t)
    @test ground_solution_[1, :, :] ≈ predict[1, :, :] rtol = 0.08
    @test ground_solution_[2, :, :] ≈ predict[2, :, :] rtol = 0.08
    @test ground_solution_ ≈ predict rtol = 0.08
end
