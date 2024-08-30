using Test
using OptimizationOptimisers
using Lux
using Statistics, Random
using NeuralOperators
# using NeuralPDE

function get_trainset(bounds, tspan, number_of_parameters, dt)
    p_ = range(start = bounds[1][1], length = number_of_parameters, stop = bounds[1][2])
    p = collect(reshape(p_, 1, size(p_, 1)))
    t_ = collect(tspan[1]:dt:tspan[2])
    t = collect(reshape(t_, 1, size(t_, 1), 1))
    (p, t)
end

@testset "Example du = cos(p * t)" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    prob = ODEProblem(equation, u0, tspan)
    deeponet = NeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))
    u = rand(1, 50)
    v = rand(1, 40, 1)
    branch = deeponet.branch
    θ, st = Lux.setup(Random.default_rng(), branch)
    b = branch(u, θ, st)[1]
    trunk = deeponet.trunk
    θ, st = Lux.setup(Random.default_rng(), trunk)
    t = trunk(v, θ, st)[1]
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    deeponet((u, v), θ, st)[1]

    bounds = [(pi, 2pi)]
    number_of_parameters = 50
    strategy = StochasticTraining(40)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 2000)
    ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    dt = 0.025f0
    p, t = get_trainset(bounds, tspan, number_of_parameters, dt)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
    p, t = get_trainset(bounds, tspan, 100, 0.01)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
end

@testset "Example du = cos(p * t) + u" begin
    eq_(u, p, t) = cos(p * t) + u
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    prob = ODEProblem(eq_, u0, tspan)
    deeponet = NeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))
    bounds = [(0.1f0, 2.0f0)]
    number_of_parameters = 40
    dt = (tspan[2] - tspan[1]) / 40
    strategy = StochasticTraining(50)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 4000)
    sol.original.objective
    #if u0 == 1
    ground_analytic_(u0, p, t) = (p * sin(p * t) - cos(p * t) + (p^2 + 2) * exp(t)) /
                                 (p^2 + 1)
    p, t = get_trainset(bounds, tspan, number_of_parameters, dt)
    ground_solution = ground_analytic_.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.5
end

@testset "Example with data du = p*t^2" begin
    equation = (u, p, t) -> p * t^2
    tspan = (0.0f0, 1.0f0)
    u0 = 0.0f0
    prob = ODEProblem(equation, u0, tspan)
    deeponet = NeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))
    bounds = [(0.0f0, 10.0f0)]
    number_of_parameters = 60
    dt = (tspan[2] - tspan[1]) / 40
    strategy = StochasticTraining(60)
    opt = OptimizationOptimisers.Adam(0.03)
    #generate data
    ground_analytic = (u0, p, t) -> u0 + p * t^3 / 3
    function get_data()
        sol = ground_analytic.(u0, p, vec(t))
        tuple_ = (p, t)
        sol, tuple_
    end
    u = rand(1, 50)
    v = rand(1, 40, 1)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet((u, v), θ, st)[1]
    data, tuple_ = get_data()
    function additional_loss_(phi, θ)
        u = phi(tuple_, θ)
        norm = prod(size(u))
        sum(abs2, u .- data) / norm
    end
    alg = PINOODE(
        deeponet, opt, bounds, number_of_parameters; strategy = strategy,
        additional_loss = additional_loss_)
    sol = solve(prob, alg, verbose = false, maxiters = 2000)

    p, t = get_trainset(bounds, tspan, number_of_parameters, dt)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
end

#multiple parameters
@testset "Example du = cos(p * t)" begin
    equation = (u, p, t) -> p[1] * cos(p[2] * t) + p[3]
    tspan = (0.0, 1.0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)

    input_branch_size = 3
    deeponet = NeuralOperators.DeepONet(
        Chain(
            Dense(input_branch_size => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))

    u = rand(3, 50)
    v = rand(1, 40, 1)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet((u, v), θ, st)[1]

    bounds = [(1.0, pi), (1.0, 2.0), (2.0, 3.0)]
    number_of_parameters = 50
    strategy = StochasticTraining(70)
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 3000)

    function get_trainset(bounds, tspan, number_of_parameters, dt)
        p_ = [range(start = b[1], length = number_of_parameters, stop = b[2])
              for b in bounds]
        p = vcat([collect(reshape(p_i, 1, size(p_i, 1))) for p_i in p_]...)
        t_ = collect(tspan[1]:dt:tspan[2])
        t = collect(reshape(t_, 1, size(t_, 1), 1))
        (p, t)
    end

    ground_solution = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t) + p[3] * t
    function ground_solution_f(p, t)
        reduce(hcat,
            [[ground_solution(u0, p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
    end

    (p, t) = get_trainset(bounds, tspan, 50, 0.025f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp((p, t))
    @test ground_solution_≈predict rtol=0.05

    p, t = get_trainset(bounds, tspan, 100, 0.01f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp((p, t))
    @test ground_solution_≈predict rtol=0.05
end
