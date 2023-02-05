using NeuralPDE, OrdinaryDiffEq, DiffEqFlux, OptimizationPolyalgorithms, Lux, Random, Flux

function f(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
true_sol = solve(prob_oop, Tsit5())

func = sin
N = 25
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                    Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                    Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))

opt = BFGS(initial_stepnorm = 0.01)
θ = Lux.setup(Random.default_rng(), chain)[1] |> Lux.ComponentArray .|> Float64
alg = NeuralPDE.NNODE(chain, opt, θ, autodiff = false, strategy = GridTraining(0.01))
sol = solve(prob_oop, alg, verbose=true, maxiters = 2000, abstol = 1e-8)

using Plots

plot(sol)
plot!(true_sol)
ylims!(0,8)