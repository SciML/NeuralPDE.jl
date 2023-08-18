
using OrdinaryDiffEq, OptimizationPolyalgorithms, Lux, OptimizationOptimJL, Test, Statistics, Plots, Optimisers

function fu(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
tspan = (0.0, 3.0)
points1 = [rand() for i=1:140]
points2 = [rand() + 1 for i=1:40]
points3 = [rand() + 2 for i=1:20]
addedPoints = vcat(points1, points2, points3)

prob_oop = ODEProblem{false}(fu, u0, tspan, p)
true_sol = solve(prob_oop, Tsit5(), saveat = 0.01)
func = Lux.σ
N = 12
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                    Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))

opt = Optimisers.Adam(0.01)

#bad choices for weights, samples and dx so that the algorithm will fail without the added points
weights = [0.3, 0.3, 0.4]
samples = 3
dx = 1.0

#Grid Training without added points (difference between solutions should be high)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.GridTraining(dx))
sol = solve(prob_oop, alg, verbose=true, maxiters = 20000, saveat = 0.01)

@test abs(mean(sol) - mean(true_sol)) > 0.2

#Grid Training with added points (difference between solutions should be low)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.GridTraining(dx))
sol = solve(prob_oop, alg, verbose=true, maxiters = 20000, saveat = 0.01, tstops = addedPoints)

@test abs(mean(sol) - mean(true_sol)) < 0.2

#WeightedIntervalTraining without added points (difference between solutions should be high)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.WeightedIntervalTraining(weights, samples))
sol = solve(prob_oop, alg, verbose=true, maxiters = 20000, saveat = 0.01)

@test abs(mean(sol) - mean(true_sol)) > 0.2

#WeightedIntervalTraining with added points (difference between solutions should be low)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.WeightedIntervalTraining(weights, samples))
sol = solve(prob_oop, alg, verbose=true, maxiters = 20000, saveat = 0.01, tstops = addedPoints)

@test abs(mean(sol) - mean(true_sol)) < 0.2

#StochasticTraining without added points (difference between solutions should be high)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.StochasticTraining(samples))
sol = solve(prob_oop, alg, verbose=true, maxiters = 20000, saveat = 0.01)

@test abs(mean(sol) - mean(true_sol)) > 0.2

#StochasticTraining with added points (difference between solutions should be low)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.StochasticTraining(samples))
sol = solve(prob_oop, alg, verbose=true, maxiters = 20000, saveat = 0.01, tstops = addedPoints)

@test abs(mean(sol) - mean(true_sol)) < 0.2

using Plots

plot(sol)
plot!(true_sol)
ylims!(0,8)
