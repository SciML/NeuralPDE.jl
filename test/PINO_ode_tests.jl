using Test
using OrdinaryDiffEq,OptimizationOptimisers
using Flux, Lux
using Statistics, Random
using NeuralOperators
# using NeuralPDE

@testset "Example 1" begin
    linear_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
    linear = (u, p, t) -> cos(p * t)
    tspan = (0.0, 2.0)
    u0 = 0.0
    a = Float32.([i  for i in 0.1:0.05:pi])

    u_output_ = []
    for i in 1:length(a)
        prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan, a[i])
        sol1 = solve(prob, Tsit5(); saveat = 0.02)
        push!(u_output_, Float32.(sol1.u'))
    end
    #TODO u0 -> [u0,a]?
    prob = ODEProblem(linear, u0, tspan)
    chain = Lux.Chain(Lux.Dense(2, 5, Lux.σ), Lux.Dense(5, 1))
    # opt = OptimizationOptimisers.Adam(0.1)
    opt = OptimizationOptimisers.Adam(0.03)
    training_mapping = (a, u_output_)
    alg = NeuralPDE.PINOODE(chain, opt, training_mapping)

    sol = solve(prob,
        alg, verbose = true,
        maxiters = 2000, abstol = 1.0f-7)

    total_loss
    total_loss, optprob, opt, callback, maxiters = sol
    phi(rand(2, 10), init_params)

    total_loss(init_params, 1)

     using Plots
end
