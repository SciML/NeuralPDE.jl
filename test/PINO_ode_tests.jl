using Test
using OrdinaryDiffEq, OptimizationOptimisers
using Flux, Lux
using Statistics, Random
using NeuralOperators
using NeuralPDE

@testset "Example 1" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0, 2.0)
    u0 = 2.0
    #generate data set
    t0, t_end = tspan
    instances_size = 100
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    as = [i for i in range(0.1, stop = pi / 2, length = batch_size)]
    # # p = ps TODO
    # patamaters_set = []
    # for a_i in as
    #     a_arr = fill(a_i, instances_size)'
    #     t_and_p = Float32.(reshape(reduce(vcat, [ts, a_arr]), 2, instances_size, 1))
    #     push!(patamaters_set, t_and_p)
    # end

    u_output_ = Array{Float32, 3}[]
    prob_set = []
    for a_i in as
        prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, a_i)
        sol1 = solve(prob, Tsit5(); saveat = 0.0204)
        reshape_sol = Float32.(reshape(sol1(range_).u', 1, instances_size, 1))
        push!(prob_set, prob)
        push!(u_output_, reshape_sol)
    end

    """
    Set of training data:
    * input data: mesh of 't' paired with set of parameters 'a':
    * output data: set of corresponding parameter 'a' solutions u(t){a}
     """
    train_set = TRAINSET(prob_set, u_output_);
    #TODO u0 ?
    prob = ODEProblem(linear, u0, tspan)
    chain = Lux.Chain(Lux.Dense(2, 20, Lux.σ), Lux.Dense(20, 20, Lux.σ), Lux.Dense(20, 1));
    opt = OptimizationOptimisers.Adam(0.03)

    alg = PINOODE(chain, opt, train_set);

    res, phi = solve(prob,
        alg, verbose = true,
        maxiters = 2000, abstol = 1.0f-10)

    predict = reduce(vcat,
        [phi(reduce(vcat, [ts, fill(train_set.input_data[i].p, 1, size(ts)[2])]), res.u)
         for i in 1:batch_size])
    ground = reduce(vcat, [train_set.output_data[i] for i in 1:batch_size])
    @test ground≈predict atol=0.5
end

@testset "Example 2" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0, 2.0)
    p = pi
    u0 = 2
    #generate data set
    t0, t_end = tspan
    instances_size = 100
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    u0s = [i for i in range(0.0, stop = pi / 2, length = batch_size)]
    # initial_condition_set = []
    # u0_arr = []
    # for u0_i in u0s
    #     u0_i_arr = reshape(fill(u0_i, instances_size)', 1, instances_size, 1)
    #     push!(u0_arr, u0_i_arr)
    #     t_and_u0 = reshape(reduce(vcat, [ts, u0_i_arr]), 2, instances_size, 1)
    #     push!(initial_condition_set, t_and_u0)
    # end

    u_output_ = Array{Float32, 3}[]
    prob_set = []
    for u0_i in u0s
        prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0_i, tspan, p)
        sol1 = solve(prob, Tsit5(); saveat = 0.0204)
        reshape_sol = reshape(sol1(range_).u', 1, instances_size, 1)
        push!(prob_set, prob)
        push!(u_output_, reshape_sol)
    end

    """
      Set of training data:
      * input data: mesh of 't' paired with set of initial conditions 'a':
      * output data: set of corresponding parameter 'a' solutions u(t){a}
    """
    train_set = TRAINSET(prob_set, u_output_; isu0 = true);
    #TODO u0 ?
    prob = ODEProblem(linear, u0, tspan, p)
    chain = Lux.Chain(Lux.Dense(2, 20, Lux.σ), Lux.Dense(20, 20, Lux.σ), Lux.Dense(20, 1));
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(chain, opt, train_set);
    res, phi = solve(prob,
        alg, verbose = true,
        maxiters = 2000, abstol = 1.0f-10)

    predict = reduce(vcat,
        [phi(reduce(vcat, [ts, fill(train_set.input_data[i].u0, 1, size(ts)[2])]), res.u)
         for i in 1:batch_size])
    ground = reduce(vcat, [train_set.output_data[i] for i in 1:batch_size])
    @test ground≈predict atol=0.5
end

@testset "lotka volterra" begin
    function lotka_volterra(u, p, t)
        # Model parameters.
        α, β, γ, δ = p
        # Current state.
        x, y = u
        # Evaluate differential equations.
        dx = (α - β * y) * x # prey
        dy = (δ * x - γ) * y # predator
        return [dx, dy]
    end
    u0 = [1.0, 1.0]
    p = [1.5, 1.0, 3.0, 1.0]
    tspan = (0.0, 4.0)
    dt = 0.01

    instances_size = 100
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    ps = [p .+ i*[0.015, 0.01, 0.03, 0.01] for i in 1:batch_size]

    u_output_ = []
    prob_set = []
    plot_set =[]
    for p_i in ps
        prob = ODEProblem(lotka_volterra, u0, tspan, p_i)
        solution = solve(prob, Tsit5(); saveat = dt)
        reshape_sol = Float32.(reduce(hcat,solution(range_).u))
        push!(plot_set,solution)
        push!(prob_set, prob)
        push!(u_output_, reshape_sol)
    end

    train_set = TRAINSET(prob_set, u_output_);
    #TODO u0 ?
    prob = ODEProblem(linear, u0, tspan, p)
    chain = Lux.Chain(Lux.Dense(5, 20, Lux.σ), Lux.Dense(20, 20, Lux.σ), Lux.Dense(20, 2))
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(chain, opt, train_set);
    res, phi = solve(prob,
        alg, verbose = true,
        maxiters = 2000, abstol = 1.0f-10)

    predict = reduce(vcat,
        [phi(reduce(vcat,
                [ts, reduce(hcat, fill(train_set.input_data[i].p, 1, size(ts)[2]))]),
            res.u)
         for i in 1:batch_size])
    ground = reduce(vcat, [train_set.output_data[i] for i in 1:batch_size])
    @test ground≈predict atol=5
end
