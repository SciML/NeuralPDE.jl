
@testsetup module PINOODETestSetup
    using OptimizationOptimisers
    using Lux
    using NeuralOperators
    using NeuralPDE

    function get_trainset(chain::DeepONet, bounds, number_of_parameters, tspan, dt)
        p_ = [range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]
        p = vcat([collect(reshape(p_i, 1, size(p_i, 1))) for p_i in p_]...)
        t_ = collect(tspan[1]:dt:tspan[2])
        t = reshape(t_, 1, size(t_, 1), 1)
        (p, t)
    end

    function get_trainset(chain::Lux.Chain, bounds, number_of_parameters, tspan, dt)
        tspan_ = tspan[1]:dt:tspan[2]
        pspan = [range(start = b[1], length = number_of_parameters, stop = b[2])
                for b in bounds]
        x_ = hcat(vec(map(
            points -> collect(points), Iterators.product([pspan..., tspan_]...)))...)
        x = reshape(x_, size(bounds, 1) + 1, prod(size.(pspan, 1)), size(tspan_, 1))
        p, t = x[1:(end - 1), :, :], x[[end], :, :]
        (p, t)
    end
    export get_trainset
end
#Test Chain with Float64 accuracy
@testitem "Example du = cos(p * t)" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)
    chain = Chain(
        Dense(2 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 1))
    x = rand(2, 50, 10)
    θ, st = Lux.setup(Random.default_rng(), chain)
    b = chain(x, θ, st)[1]

    bounds = [(pi, 2pi)]
    number_of_parameters = 300
    strategy = StochasticTraining(300)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(
        chain, opt, bounds, number_of_parameters; strategy = strategy, init_params = θ |>
                                                                                     f64)
    sol = solve(prob, alg, verbose = false, maxiters = 5000)
    ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    dt = 0.025f0
    p, t = get_trainset(chain, bounds, number_of_parameters, tspan, dt)
    ground_solution = ground_analytic.(u0, p, t)
    predict_sol = sol.interp(reduce(vcat, (p, t)))
    @test ground_solution≈predict_sol rtol=0.05
    p, t = get_trainset(chain, bounds, 100, tspan, 0.01)
    ground_solution = ground_analytic.(u0, p, t)
    predict_sol = sol.interp(reduce(vcat, (p, t)))
    @test ground_solution≈predict_sol rtol=0.05
end

#Test DeepONet with Float64 accuracy
@testitem "Example du = cos(p * t)" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)
    deeponet = NeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))
    u = rand(Float32, 1, 50)
    v = rand(Float32, 1, 40, 1)
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
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters;
        strategy = strategy, init_params = θ |> f64)
    sol = solve(prob, alg, verbose = false, maxiters = 3000)
    ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    dt = 0.025f0
    p, t = get_trainset(deeponet, bounds, number_of_parameters, tspan, dt)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
    p, t = get_trainset(deeponet, bounds, 100, tspan, 0.01)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
end

@testitem "Example du = cos(p * t) + u" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
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
    strategy = GridTraining(0.1f0)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 4000)
    sol.original.objective
    #if u0 == 1
    ground_analytic_(u0, p, t) = (p * sin(p * t) - cos(p * t) + (p^2 + 2) * exp(t)) /
                                 (p^2 + 1)
    p, t = get_trainset(deeponet, bounds, number_of_parameters, tspan, dt)
    ground_solution = ground_analytic_.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
end

@testitem "Example with data du = p*t^2" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
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
    p, t = get_trainset(deeponet, bounds, number_of_parameters, tspan, dt)
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

    p, t = get_trainset(deeponet, bounds, number_of_parameters, tspan, dt)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))
    @test ground_solution≈predict_sol rtol=0.05
end

#multiple parameters Сhain
@testitem "Example multiple parameters Сhain du = p1 * cos(p2 * t) + p3" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
    equation = (u, p, t) -> p[1] * cos(p[2] * t) + p[3]
    tspan = (0.0, 1.0)
    u0 = 1.0
    prob = ODEProblem(equation, u0, tspan)

    input_branch_size = 3
    chain = Chain(
        Dense(input_branch_size + 1 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast), Dense(10 => 1))

    x = rand(Float32, 4, 1000, 10)
    θ, st = Lux.setup(Random.default_rng(), chain)
    c = chain(x, θ, st)[1]

    bounds = [(1.0, pi), (1.0, 2.0), (2.0, 3.0)]
    number_of_parameters = 200
    strategy = StochasticTraining(200)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(chain, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 4000)

    ground_solution = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t) + p[3] * t

    function ground_solution_f(p, t)
        reduce(hcat,
            [[ground_solution(u0, p[:, i, j], t[1, i, j]) for j in axes(t, 3)]
             for i in axes(p, 2)])'
    end
    (p, t) = get_trainset(chain, bounds, 20, tspan, 0.1f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp(reduce(vcat, (p, t)))[1, :, :]
    @test ground_solution_≈predict rtol=0.05

    p, t = get_trainset(chain, bounds, 50, tspan, 0.025f0)
    ground_solution_ = ground_solution_f(p, t)
    predict_sol = sol.interp(reduce(vcat, (p, t)))[1, :, :]
    @test ground_solution_≈predict_sol rtol=0.05
end

#multiple parameters DeepOnet
@testitem "Example multiple parameters DeepOnet du = p1 * cos(p2 * t) + p3" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
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
    strategy = StochasticTraining(20)
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 4000)
    ground_solution = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t) + p[3] * t
    function ground_solution_f(p, t)
        reduce(hcat,
            [[ground_solution(u0, p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
    end

    (p, t) = get_trainset(deeponet, bounds, 50, tspan, 0.025f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp((p, t))
    @test ground_solution_≈predict rtol=0.05

    p, t = get_trainset(deeponet, bounds, 100, tspan, 0.01f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp((p, t))
    @test ground_solution_≈predict rtol=0.05
end

#vector output
@testitem "Example du = [cos(p * t), sin(p * t)]" tags=[:pinoode] setup=[PINOODETestSetup] begin
    using NeuralPDE, Lux, OptimizationOptimisers, NeuralOperators
    equation = (u, p, t) -> [cos(p * t), sin(p * t)]
    tspan = (0.0f0, 1.0f0)
    u0 = [1.0f0, 0.0f0]
    prob = ODEProblem(equation, u0, tspan)
    input_branch_size = 1
    chain = Chain(
        Dense(input_branch_size + 1 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast), Dense(10 => 2))
    bounds = [(pi, 2pi)]
    number_of_parameters = 300
    strategy = StochasticTraining(300)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(chain, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 6000)

    ground_solution = (u0, p, t) -> [1 + sin(p * t) / p, 1 / p - cos(p * t) / p]
    function ground_solution_f(p, t)
        ans_1 = reduce(hcat,
            [reduce(vcat,
                 [ground_solution(u0, p[1, i, 1], t[1, 1, j])[1] for i in axes(p, 2)])
             for j in axes(t, 3)])
        ans_2 = reduce(hcat,
            [reduce(vcat,
                 [ground_solution(u0, p[1, i, 1], t[1, 1, j])[2] for i in axes(p, 2)])
             for j in axes(t, 3)])

        ans_1 = reshape(ans_1, 1, size(ans_1)...)
        ans_2 = reshape(ans_2, 1, size(ans_2)...)
        vcat(ans_1, ans_2)
    end
    p, t = get_trainset(chain, bounds, 50, tspan, 0.01f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp(reduce(vcat, (p, t)))
    @test ground_solution_[1, :, :]≈predict[1, :, :] rtol=0.05
    @test ground_solution_[2, :, :]≈predict[2, :, :] rtol=0.05
    @test ground_solution_≈predict rtol=0.05

    p, t = get_trainset(chain, bounds, 300, tspan, 0.01f0)
    ground_solution_ = ground_solution_f(p, t)
    predict = sol.interp(reduce(vcat, (p, t)))
    @test ground_solution_[1, :, :]≈predict[1, :, :] rtol=0.05
    @test ground_solution_[2, :, :]≈predict[2, :, :] rtol=0.05
    @test ground_solution_≈predict rtol=0.3
end
