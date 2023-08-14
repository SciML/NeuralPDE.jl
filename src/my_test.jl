
using OrdinaryDiffEq, OptimizationPolyalgorithms, Lux, OptimizationOptimJL, Test, Statistics, Plots, Optimisers

function fu(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob_oop = ODEProblem{false}(fu, u0, (0.0, 3.0), p)
true_sol = solve(prob_oop, Tsit5(), saveat = 0.01)
func = Lux.σ
N = 12
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, length(u0)))

opt = Optimisers.Adam(0.01)
dx=0.05
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.GridTraining(dx))
sol = solve(prob_oop, alg, verbose=true, maxiters = 3, saveat = 0.01)

@test abs(mean(sol) - mean(true_sol)) < 0.2

# using Plots

# plot(sol)
# plot!(true_sol)
# ylims!(0,8)
