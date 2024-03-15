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

    """
    Set of training data:
    * input data: set of parameters 'a':
    * output data: set of solutions u(t){a} corresponding parameter 'a'
    """
    train_set = NeuralPDE.TRAINSET(prob_set, u_output_)

    #TODO u0 ?
    prob = ODEProblem(linear, u0, tspan, 0)
    inner = 20
    chain = Lux.Chain(Lux.Dense(2, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, inner, Lux.σ),
        Lux.Dense(inner, 1))
    ps = Lux.setup(Random.default_rng(), chain)[1] |> ComponentArray |> gpud

    flat_no = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, 1), modes = (16,),
        σ = gelu)

    opt = OptimizationOptimisers.Adam(0.03)
    alg = NeuralPDE.PINOODE(flat_no, opt, train_set, ps)
    pino_solution = solve(prob, alg, verbose = true, maxiters = 200)
    predict = pino_solution.predict |> cpu
    ground = u_output_
    @test ground≈predict atol=1
end
