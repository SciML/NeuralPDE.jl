using Test
using OrdinaryDiffEq, OptimizationOptimisers
using Lux
using Statistics, Random
using NeuralOperators
using NeuralPDE

@testset "Example p" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    u0 = 0.0f0
    #generate data set
    t0, t_end = tspan
    instances_size = 50
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    as = [Float32(i) for i in range(0.1, stop = pi / 2, length = batch_size)]

    u_output_ = zeros(Float32, 1, instances_size, batch_size)
    prob_set = []
    for (i, a_i) in enumerate(as)
        prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, a_i)
        sol1 = solve(prob, Tsit5(); saveat = 0.0204)
        reshape_sol = Float32.(reshape(sol1(range_).u', 1, instances_size, 1))
        push!(prob_set, prob)
        u_output_[:, :, i] = reshape_sol
    end

    """
    Set of training data:
    * input data: set of parameters 'a':
    * output data: set of solutions u(t){a} corresponding parameter 'a'
     """
train_set = TRAINSET(prob_set, u_output_);
    #TODO u0 ?
    prob = ODEProblem(linear, u0, tspan, 0)
    chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ),
    Lux.Dense(16, 16, Lux.σ),
    Lux.Dense(16, 16, Lux.σ),
    Lux.Dense(16, 16, Lux.σ),
    Lux.Dense(16, 32, Lux.σ),
    Lux.Dense(32, 1))
    flat_no = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,),
        σ = gelu)
    η₀ = 1.0f-2
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(flat_no, opt, train_set)
    res, phi = solve(prob, alg, verbose = true, maxiters = 200)

    input_data_set = Array{Float32, 3}(undef, 2, instances_size, batch_size)
    for (i, prob) in enumerate(prob_set)
        in_ = reduce(vcat, [ts, fill(prob.p, 1, size(ts)[2], 1)])
        input_data_set[:, :, i] = in_
    end
    predict = phi(input_data_set, res.u)
    ground = u_output_
    @test ground≈predict atol=1
end

"Example u0"
begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    p = Float32(pi)
    u0 = 2.0f0
    #generate data set
    t0, t_end = tspan
    instances_size = 50
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    u0s = [i for i in range(0.0f0, stop = pi / 2.0f0, length = batch_size)]
    u_output_ = zeros(Float32, 1, instances_size, batch_size)
    prob_set = []
    for (i, u0_i) in enumerate(u0s)
        prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0_i, tspan, p)
        sol1 = solve(prob, Tsit5(); saveat = 0.0204)
        reshape_sol = Float32.(reshape(sol1(range_).u', 1, instances_size, 1))
        push!(prob_set, prob)
        u_output_[:, :, i] = reshape_sol
    end

    """
      Set of training data:
      * input data: set of initial conditions 'u0':
      * output data: set of solutions u(t){u0} corresponding initial conditions 'u0'
    """
    train_set = TRAINSET(prob_set, u_output_; isu0 = true)
    #TODO u0 ?
    prob = ODEProblem(linear, 0.0f0, tspan, p)
    # chain = Lux.Chain(Lux.Dense(2, 20, Lux.σ), Lux.Dense(20, 20, Lux.σ), Lux.Dense(20, 1))
    fno = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,),
        σ = gelu)
    opt = OptimizationOptimisers.Adam(0.001)
    alg = PINOODE(fno, opt, train_set)
    res, phi = solve(prob,
        alg, verbose = true,
        maxiters = 200)

    input_data_set = Array{Float32, 3}(undef, 2, instances_size, batch_size)
    for (i, prob) in enumerate(prob_set)
        in_ = reduce(vcat, [ts, fill(prob.u0, 1, size(ts)[2], 1)])
        input_data_set[:, :, i] = in_
    end
    predict = phi(input_data_set, res.u)
    ground = u_output_
    @test ground≈predict atol=1.0
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

    u0 = [1.0f0, 1.0f0]
    p = Float32[1.5, 1.0, 3.0, 1.0]
    tspan = (0.0f0, 4.0f0)
    dt = 0.01f0

    instances_size = 100
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    ps = [p .+ i * Float32[0.000, 0.0, 0.001, 0.01] for i in 1:batch_size]
    u_output_ = zeros(Float32, 2, instances_size, batch_size)
    prob_set = []
    for (i, p_i) in enumerate(ps)
        prob = ODEProblem(lotka_volterra, u0, tspan, p_i)
        solution = solve(prob, Tsit5(); saveat = dt)
        reshape_sol_ = reduce(hcat, solution(range_).u)
        reshape_sol = Float32.(reshape(reshape_sol_, 2, instances_size, 1))
        push!(prob_set, prob)
        u_output_[:, :, i] = reshape_sol
    end

    train_set = TRAINSET(prob_set, u_output_);
    #TODO u0 ?
    prob = ODEProblem(lotka_volterra_matrix, u0, tspan, p)
    chain = Lux.Chain(Lux.Dense(5, 20, Lux.σ), Lux.Dense(20, 20, Lux.σ), Lux.Dense(20, 2))
    flat_no = FourierNeuralOperator(ch = (5, 16, 16, 16, 16, 16, 32, 2), modes = (16,),
        σ = gelu)
    opt = OptimizationOptimisers.Adam(0.01)
    alg = PINOODE(flat_no, opt, train_set);
    res, phi = solve(prob, alg, verbose = true, maxiters = 200)

    input_data_set = Array{Float32, 3}(undef, 5, instances_size, batch_size)
    for (i, prob) in enumerate(prob_set)
        inner = reduce(vcat, [ts, reduce(hcat, fill(prob.p, 1, size(ts)[2], 1))])
        in_ = reshape(inner, size(inner)..., 1)
        input_data_set[:, :, i] = in_
    end

    predict = phi(input_data_set, res.u)
    ground = u_output_

    @test ground≈predict atol=5
end
