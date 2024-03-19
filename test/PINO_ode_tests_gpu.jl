using Test
using OrdinaryDiffEq
using Lux
using ComponentArrays
using NeuralOperators
using OptimizationOptimisers
using Random
using LuxCUDA
using NeuralPDE

CUDA.allowscalar(false)
const gpud = gpu_device()

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
    u_output_ = u_output_ |> gpud

    """
    Set of training data:
    * input data: set of parameters 'a',
    * output data: set of solutions u(t){a} corresponding parameter 'a'.
    """
    train_set = TRAINSET(prob_set, u_output_)

    prob = ODEProblem(linear, u0, tspan, 0)
    inner = 50
    chain = Lux.Chain(Lux.Dense(2, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, 1))
    ps = Lux.setup(Random.default_rng(), chain)[1] |> ComponentArray |> gpud
    opt = OptimizationOptimisers.Adam(0.03)
    alg = PINOODE(chain, opt, train_set; init_params = ps)
    pino_solution = solve(prob, alg, verbose = true, maxiters = 1000)
    predict = pino_solution.predict |> cpu
    ground = u_output_ |> cpu
    @test ground≈predict atol=1
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

    train_set = TRAINSET(prob_set, u_output_)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    flat_no = FourierNeuralOperator(ch = (5, 64, 64, 64, 64, 64, 128, 2), modes = (16,),
        σ = gelu)
    flat_no = Lux.transform(flat_no)
    ps = Lux.setup(Random.default_rng(), flat_no)[1] |> ComponentArray |> gpud

    opt = OptimizationOptimisers.Adam(0.001)
    alg = PINOODE(
        flat_no, opt, train_set; init_params = ps, is_data_loss = true, is_physics_loss = true)
    pino_solution = solve(prob, alg, verbose = true, maxiters = 1000)
    predict = pino_solution.predict |> cpu
    ground = u_output_
    @test ground≈predict atol=2
end
