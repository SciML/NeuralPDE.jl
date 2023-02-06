using OrdinaryDiffEq, DiffEqFlux, OptimizationPolyalgorithms, Lux, Random, Flux

include("NeuralPDE.jl")

function f(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
true_sol = solve(prob_oop, Tsit5())

func = sin
N = 12
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                    Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))

opt = ADAM(0.01)
alg = NeuralPDE.NNODE(chain, opt, autodiff = false, strategy = NeuralPDE.FrontWeightedGridTraining(0.07))
sol = solve(prob_oop, alg, verbose=true, maxiters = 100000)

using Plots

plot(sol)
plot!(true_sol)
ylims!(0,8)