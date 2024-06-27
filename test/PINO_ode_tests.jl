using Test
using OptimizationOptimisers
using Lux
using Statistics, Random
using LuxNeuralOperators
using NeuralPDE

# dG(u(t, p), t) = f(G,u(t, p))
@testset "Example du = cos(p * t)" begin
    equation = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    prob = ODEProblem(equation, u0, tspan)
    deeponet = LuxNeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))
    u = rand(1, 50)
    v = rand(1, 40, 1)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet((u, v), θ, st)[1]

    branch = deeponet.layers.branch
    θ, st = Lux.setup(Random.default_rng(), branch)
    b = branch(u, θ, st)[1]

    trunk = deeponet.layers.trunk
    θ, st = Lux.setup(Random.default_rng(), trunk)
    t = trunk(v, θ, st)[1]

    bounds = [(pi, 2pi)]
    number_of_parameters = 50
    strategy = StochasticTraining(40)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = false, maxiters = 2000)

    ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    function get_trainset(bounds, tspan , number_of_parameters, dt)
        p_ = range(start = bounds[1][1], length = number_of_parameters, stop = bounds[1][2])
        p = collect(reshape(p_, 1, size(p_, 1)))
        t_ = collect(tspan[1]:dt:tspan[2])
        t = collect(reshape(t_, 1, size(t_, 1), 1))
        (p,t)
    end
    p,t = get_trainset(bounds, tspan, number_of_parameters, dt)

    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))

    @test ground_solution≈predict_sol rtol=0.01

    p, t = get_trainset(bounds, tspan, 100, 0.01)
    ground_solution = ground_analytic.(u0, p, vec(t))
    predict_sol = sol.interp((p, t))

    @test ground_solution≈predict_sol rtol=0.01
end

@testset "Example du = cos(p * t) + u" begin
    eq_(u, p, t) = cos(p * t) + u
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    prob = ODEProblem(eq_, u0, tspan)
    deeponet = LuxNeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))
    bounds = [(0.1f0, 2.0f0)]
    number_of_parameters = 40
    dt = (tspan[2] - tspan[1]) / 40
    strategy = GridTraining(dt)

    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)

    sol = solve(prob, alg, verbose = true, maxiters = 3000)
    sol.original.objective
    #if u0 == 1
    ground_analytic_(u0, p, t) = (p * sin(p * t) - cos(p * t) + (p^2 + 2) * exp(t)) /
                                 (p^2 + 1)

    p_ = range(start = bounds[1][1], length = number_of_parameters, stop = bounds[1][2])
    p = collect(reshape(p_, 1, size(p_,1)))
    ground_solution = ground_analytic_.(u0, p, vec(sol.t[2]))

    @test ground_solution≈sol.u rtol=0.01
end

@testset "Example with data du = p*t^2" begin
    equation = (u, p, t) -> p * t^2
    tspan = (0.0f0, 1.0f0)
    u0 = 0.0f0
    prob = ODEProblem(equation, u0, tspan)

    deeponet = LuxNeuralOperators.DeepONet(
        Chain(
            Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))

    bounds = [(0.0f0, 10.0f0)]
    number_of_parameters = 60
    dt = (tspan[2] - tspan[1]) / 40
    strategy = GridTraining(dt)

    opt = OptimizationOptimisers.Adam(0.03)

    #generate data
    ground_analytic = (u0, p, t) -> u0 + p * t^3 / 3

    function get_trainset(branch_size, trunk_size, bounds, tspan)
        p_ = range(bounds[1][1], stop = bounds[1][2], length = branch_size)
        p = reshape(p_, 1, branch_size)
        t_ = collect(range(tspan[1], stop = tspan[2], length = trunk_size))
        t = reshape(t_, 1, trunk_size, 1)
        (p, t)
    end

    function get_data()
        sol = ground_analytic.(u0, p, vec(t))
        #x = equation.(sol, p, vec(t))
        tuple_ = (p, t)
        sol, tuple_
    end

    u = rand(1, 50)
    v = rand(1, 40, 1)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet((u, v), θ, st)[1]

    branch_size, trunk_size = 50, 40
    p, t = get_trainset(branch_size, trunk_size, bounds, tspan)
    data, tuple_ = get_data()

    function additional_loss_(phi, θ)
        u = phi(tuple_, θ)
        norm = prod(size(u))
        sum(abs2, u .- data) / norm
    end

    alg = PINOODE(
        deeponet, opt, bounds, number_of_parameters; strategy = strategy,
        additional_loss = additional_loss_)
    sol = solve(prob, alg, verbose = true, maxiters = 2000)
    p_ = range(start = bounds[1][1], length = number_of_parameters, stop = bounds[1][2])
    p = reshape(p_, 1, size(p_)[1])
    ground_solution = ground_analytic.(u0, p, vec(sol.t[2]))

    @test ground_solution≈sol.u rtol=0.01
end

#vector outputs and multiple parameters
@testset "Example du = cos(p * t)" begin
    function equation1(u, p, t)
        p1, p2 = p[1], p[2]
        cos(p1 * t) + p2
    end

    equation = (u, p, t) -> p[1] * cos(p[2] * t)
    tspan = (0.0f0, 1.0f0)
    u0 = 1.0f0
    prob = ODEProblem(equation, u0, tspan)

    deeponet = LuxNeuralOperators.DeepONet(
        Chain(
            Dense(2 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
        Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
            Dense(10 => 10, Lux.tanh_fast)))

    u = rand(2, 50, 1)
    v = rand(1, 40, 1)
    θ, st = Lux.setup(Random.default_rng(), deeponet)
    c = deeponet((u, v), θ, st)[1]

    bounds = [(0.1f0, pi), (1.0f0, 2.0f0)]
    number_of_parameters = 50
    dt = (tspan[2] - tspan[1]) / 40
    strategy = GridTraining(dt)
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
    sol = solve(prob, alg, verbose = true, maxiters = 3000)

    ga = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t)
    p_ = [range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]
    p = vcat([collect(reshape(p_i, 1, size(p_i,1))) for p_i in p_]...)
    t = sol.t[2]
    ground_solution = reduce(hcat,
        [[ga(u0, p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])

    @test ground_solution≈sol.u rtol=0.01
end
