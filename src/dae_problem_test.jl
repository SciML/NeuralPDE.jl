using DAEProblemLibrary, Sundials, Optimisers, OptimizationOptimisers, DifferentialEquations
using NeuralPDE, Lux, Test, Statistics, Plots

prob = DAEProblemLibrary.prob_dae_resrob
true_sol = solve(prob, IDA(), saveat = 0.01)
# sol = solve(prob, IDA())

u0 = [1.0, 1.0, 1.0]
func = Lux.σ
N = 12
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                    Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))

opt = Optimisers.Adam(0.01)
dx = 0.05
alg = NeuralPDE.NNDAE(chain, opt, autodiff = false, strategy = NeuralPDE.GridTraining(dx))
sol = solve(prob, alg, verbose=true, maxiters = 100000, saveat = 0.01)

println(abs(mean(true_sol .- sol)))

using Plots

plot(sol)
plot!(true_sol)
# ylims!(0,8)