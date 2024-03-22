using Test
using OrdinaryDiffEq, OptimizationOptimisers
using Lux
using Statistics, Random
# using NeuralOperators
using NeuralPDE

@testset "Example p" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    u0 = 0.0f0
    p = pi / 2
    prob = ODEProblem(linear, u0, tspan, p)
    #generate data
    t0, t_end = tspan
    instances_size = 50
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    as = [Float32(i) for i in range(0.1, stop = pi / 2, length = batch_size)]

    u_output_ = zeros(Float32, 1, instances_size, batch_size)
    prob_set = []
    for (i, a_i) in enumerate(as)
        prob_ = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, a_i)
        sol1 = solve(prob_, Tsit5(); saveat = 0.0204)
        reshape_sol = Float32.(reshape(sol1(range_).u', 1, instances_size, 1))
        push!(prob_set, prob_)
        u_output_[:, :, i] = reshape_sol
    end
    """
    Training data:
    * input data: set of parameters 'a',
    * output data: set of solutions u(t){a} corresponding parameter 'a'.
    """
    train_set = TRAINSET(prob_set, u_output_)

    # operator learning phase
    # init neural network
    chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 32, Lux.σ),
        Lux.Dense(32, 32, Lux.σ),
        Lux.Dense(32, 1))
    # flat_no = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,),
    #     σ = gelu)

    opt = OptimizationOptimisers.Adam(0.01)
    pino_phase = OperatorLearning(train_set, is_data_loss = true, is_physics_loss = true)

    alg = PINOODE(chain, opt, pino_phase)
    pino_solution = solve(
        prob, alg, verbose = false, maxiters = 2000)
    predict = pino_solution.predict
    ground = u_output_
    @test ground≈predict atol=1
    # equation solving phase
    dt = (t_end - t0) / instances_size
    pino_phase = EquationSolving(dt, pino_solution)
    chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 32, Lux.σ),
        Lux.Dense(32, 1))
    alg = PINOODE(chain, opt, pino_phase)
    fine_tune_solution = solve(
        prob, alg, verbose = false, maxiters = 2000)

    fine_tune_predict = fine_tune_solution.predict
    operator_predict = pino_solution.phi(
        fine_tune_solution.input_data_set, pino_solution.res.u)
    ground = linear_analytic.(u0, p, fine_tune_solution.input_data_set[[1], :, :])
    @test ground≈fine_tune_predict atol=1.
    @test operator_predict≈fine_tune_predict rtol=0.1
end

@testset "Example u0" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0f0, 2.0f0)
    p = Float32(pi)
    u0 = 2.0f0
    prob = ODEProblem(linear, u0, tspan, p)
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
        prob_ = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0_i, tspan, p)
        sol1 = solve(prob_, Tsit5(); saveat = 0.0204)
        reshape_sol = Float32.(reshape(sol1(range_).u', 1, instances_size, 1))
        push!(prob_set, prob_)
        u_output_[:, :, i] = reshape_sol
    end
    # operator learning phase
    """
      Set of training data:
      * input data: set of initial conditions 'u0'
      * output data: set of solutions u(t){u0} corresponding initial conditions 'u0'.
    """
    train_set = TRAINSET(prob_set, u_output_)
    # fno = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,), σ = gelu)
    chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 32, Lux.σ),
        Lux.Dense(32, 32, Lux.σ),
        Lux.Dense(32, 1))
    opt = OptimizationOptimisers.Adam(0.001)
    pino_phase = OperatorLearning(train_set)
    alg = PINOODE(chain, opt, pino_phase; isu0 = true)
    pino_solution = solve(prob, alg, verbose = false, maxiters = 2000)
    predict = pino_solution.predict
    ground = u_output_
    @test ground≈predict atol=1.

    # equation solving phase
    dt = (t_end -t0) / instances_size
    pino_phase = EquationSolving(dt, pino_solution)
    chain = Lux.Chain(Lux.Dense(2, 16, Lux.σ),
        Lux.Dense(16, 16, Lux.σ),
        Lux.Dense(16, 32, Lux.σ),
        Lux.Dense(32, 1))
    alg = PINOODE(chain, opt, pino_phase; isu0 = true)
    fine_tune_solution = solve(prob, alg, verbose = false, maxiters = 2000)

    fine_tune_predict = fine_tune_solution.predict
    operator_predict = pino_solution.phi(
        fine_tune_solution.input_data_set, pino_solution.res.u)
    ground_fine_tune = linear_analytic.(u0, p, fine_tune_solution.input_data_set[[1], :, :])
    @test ground_fine_tune≈fine_tune_predict atol=1
    @test operator_predict≈fine_tune_predict rtol=0.1
end
