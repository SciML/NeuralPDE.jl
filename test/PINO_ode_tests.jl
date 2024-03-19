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
    * input data: set of parameters 'a'
    * output data: set of solutions u(t){a} corresponding parameter 'a'.
     """
    train_set = TRAINSET(prob_set, u_output_);
    prob = ODEProblem(linear, u0, tspan, 0)
    chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 32, Lux.σ),
        Lux.Dense(32, 1))
    # flat_no = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,),
    #     σ = gelu)
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(chain, opt, train_set; is_data_loss = true, is_physics_loss = true)
    pino_solution = solve(prob, alg, verbose = false, maxiters = 2000)
    predict = pino_solution.predict
    ground = u_output_
    @test ground≈predict atol=1
end

@testset "Example u0" begin
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
      * input data: set of initial conditions 'u0'
      * output data: set of solutions u(t){u0} corresponding initial conditions 'u0'.
    """
    train_set = TRAINSET(prob_set, u_output_; isu0 = true)
    #TODO we argument u0 but dont actually use u0 because we use only set of u0 for generate train set from prob_set
    prob = ODEProblem(linear, 0.0f0, tspan, p)
    fno = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,), σ = gelu)
    opt = OptimizationOptimisers.Adam(0.001)
    alg = PINOODE(fno, opt, train_set)
    pino_solution = solve(prob, alg, verbose = true, maxiters = 200)
    predict = pino_solution.predict
    ground = u_output_
    @test ground≈predict atol=1.0
end
