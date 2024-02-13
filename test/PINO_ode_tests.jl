using Test
using OrdinaryDiffEq,OptimizationOptimisers
using Flux, Lux
using Statistics, Random
using NeuralOperators
using NeuralPDE

@testset "Example 1" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0, 2.0)
    u0 = 0.0

    #generate data set
    t0, t_end = tspan
    instances_size = 100
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    batch_size = 50
    as = [i for i in range(0.1, stop = pi/2, length = batch_size)]

    patamaters_set = []
    for a_i in as
        a_arr = fill(a_i, instances_size)'
        s = reduce(vcat, [ts, a_arr])
        push!(patamaters_set, s)
    end

    u_output_ = []
    for a_i in as
        prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, a_i)
        sol1 = solve(prob, Tsit5(); saveat = 0.0204)
        push!(u_output_, sol1(range_).u')
    end

    """
    Set of training data:
    * input data: mesh of 't' paired with set of parameters 'a':
    * output data: set of corresponding parameter 'a' solutions u(t){a}
     """
    training_mapping = (patamaters_set, u_output_)

    #TODO u0 -> [u0,a]?
    prob = ODEProblem(linear, u0, tspan)
    chain = Lux.Chain(Lux.Dense(2, 20, Lux.σ), Lux.Dense(20, 20, Lux.σ), Lux.Dense(20, 1))
    opt = OptimizationOptimisers.Adam(0.03)

    alg = NeuralPDE.PINOODE(chain, opt, training_mapping)

    res, phi = solve(prob,
        alg, verbose = true,
        maxiters = 4000, abstol = 1.0f-10)


    predict = reduce(hcat, [phi(patamaters_set[i], res.u)' for i in 1:batch_size])
    grpound = reduce(hcat, [u_output_[i]' for i in 1:batch_size])
    @test grpound≈predict atol=3

    # i = 30
    # plot(predict[:, i])
    # plot!(grpound[:, i])
end
