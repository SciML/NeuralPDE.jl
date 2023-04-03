using Test, Flux
using Random, NeuralPDE
using OrdinaryDiffEq, Optimisers, Statistics
import Lux, OptimizationOptimisers, OptimizationOptimJL

Random.seed!(100)

# Run a solve on scalars
linear = (u, p, t) -> cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)

@test_throws Any solve(prob, NeuralPDE.NNODE(chain, opt; autodiff = true), dt = 1 / 20.0f0,
                       verbose = true, abstol = 1.0f-10, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true,
            abstol = 1.0f-6, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)

@test_throws Any solve(prob, NeuralPDE.NNODE(luxchain, opt; autodiff = true),
                       dt = 1 / 20.0f0,
                       verbose = true, abstol = 1.0f-10, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), verbose = true,
            abstol = 1.0f-6, maxiters = 200)

opt = OptimizationOptimJL.BFGS()
sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true,
            abstol = 1.0f-6, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), verbose = true,
            abstol = 1.0f-6, maxiters = 200)

# Run a solve on vectors
linear = (u, p, t) -> [cos(2pi * t)]
tspan = (0.0f0, 1.0f0)
u0 = [0.0f0]
prob = ODEProblem(linear, u0, tspan)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
luxchain = Lux.Chain(Lux.Dense(1, 5, σ), Lux.Dense(5, 1))

opt = OptimizationOptimJL.BFGS()
sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, abstol = 1e-10,
            verbose = true, maxiters = 200)

@test_throws Any solve(prob, NeuralPDE.NNODE(chain, opt; autodiff = true), dt = 1 / 20.0f0,
                       abstol = 1e-10, verbose = true, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), abstol = 1.0f-6,
            verbose = true, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), dt = 1 / 20.0f0, abstol = 1e-10,
            verbose = true, maxiters = 200)

@test_throws Any solve(prob, NeuralPDE.NNODE(luxchain, opt; autodiff = true),
                       dt = 1 / 20.0f0,
                       abstol = 1e-10, verbose = true, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), abstol = 1.0f-6,
            verbose = true, maxiters = 200)

@test sol(0.5) isa Vector
@test sol(0.5; idxs = 1) isa Number
@test sol.k isa SciMLBase.OptimizationSolution

#Example 1
linear = (u, p, t) -> @. t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
                         u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3)))
linear_analytic = (u0, p, t) -> [exp(-(t^2) / 2) / (1 + t + t^3) + t^2]
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), [1.0f0], (0.0f0, 1.0f0))
chain = Flux.Chain(Dense(1, 128, σ), Dense(128, 1))
luxchain = Lux.Chain(Lux.Dense(1, 128, σ), Lux.Dense(128, 1))
opt = OptimizationOptimisers.Adam(0.01)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

@test_throws Any solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true,
                       maxiters = 400)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

@test_throws Any solve(prob, NeuralPDE.NNODE(luxchain, opt; batch = true), verbose = true,
                       maxiters = 400)

sol = solve(prob,
            NeuralPDE.NNODE(chain, opt; batch = false, strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

sol = solve(prob,
            NeuralPDE.NNODE(chain, opt; batch = true, strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

sol = solve(prob,
            NeuralPDE.NNODE(luxchain, opt; batch = false,
                            strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

sol = solve(prob,
            NeuralPDE.NNODE(luxchain, opt; batch = true,
                            strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = false), verbose = true,
            maxiters = 400, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true, maxiters = 400,
            dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt; batch = false), verbose = true,
            maxiters = 400, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt; batch = true), verbose = true,
            maxiters = 400,
            dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

#Example 2
linear = (u, p, t) -> -u / 5 + exp(-t / 5) .* cos(t)
linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
luxchain = Lux.Chain(Lux.Dense(1, 5, σ), Lux.Dense(5, 1))

opt = OptimizationOptimisers.Adam(0.1)
sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

@test_throws Any solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true,
                       maxiters = 400,
                       abstol = 1.0f-8)

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt), verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

@test_throws Any solve(prob, NeuralPDE.NNODE(luxchain, opt; batch = true), verbose = true,
                       maxiters = 400,
                       abstol = 1.0f-8)

sol = solve(prob,
            NeuralPDE.NNODE(chain, opt; batch = false, strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

sol = solve(prob,
            NeuralPDE.NNODE(chain, opt; batch = true, strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

sol = solve(prob,
            NeuralPDE.NNODE(luxchain, opt; batch = false,
                            strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

sol = solve(prob,
            NeuralPDE.NNODE(luxchain, opt; batch = true,
                            strategy = StochasticTraining(100)),
            verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = false), verbose = true,
            maxiters = 400,
            abstol = 1.0f-8, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true, maxiters = 400,
            abstol = 1.0f-8, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt; batch = false), verbose = true,
            maxiters = 400,
            abstol = 1.0f-8, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(luxchain, opt; batch = true), verbose = true,
            maxiters = 400,
            abstol = 1.0f-8, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

# WeightedIntervalTraining(Lux Chain)
function f(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
true_sol = solve(prob_oop, Tsit5(), saveat = 0.01)
func = Lux.σ
N = 12
chain = Lux.Chain(Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
                  Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))

opt = Optimisers.Adam(0.01)
weights = [0.7, 0.2, 0.1]
samples = 200
alg = NeuralPDE.NNODE(chain, opt, autodiff = false,
                      strategy = NeuralPDE.WeightedIntervalTraining(weights, samples))
sol = solve(prob_oop, alg, verbose = true, maxiters = 100000, saveat = 0.01)

@test abs(mean(sol) - mean(true_sol)) < 0.2

# Checking if additional_loss feature works for NNODE
linear = (u, p, t) -> cos(2pi * t)
linear_analytic = (u, p, t) -> (1 / (2pi)) * sin(2pi * t)
tspan = (0.0f0, 1.0f0)
dt = (tspan[2] - tspan[1]) / 99
ts = collect(tspan[1]:dt:tspan[2])
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))
opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

# Analytical solution
u_analytical(x) = (1 / (2pi)) .* sin.(2pi .* x)

# GridTraining (Flux Chain)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))

(u_, t_) = (u_analytical(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(chain, opt, strategy = GridTraining(0.01),
                       additional_loss = additional_loss)

sol1 = solve(prob, alg1, verbose = true, abstol = 1.0f-8, maxiters = 500)
@test sol1.errors[:l2] < 0.5

# GridTraining (Lux Chain)
luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

(u_, t_) = (u_analytical(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(luxchain, opt, strategy = GridTraining(0.01),
                       additional_loss = additional_loss)

sol1 = solve(prob, alg1, verbose = true, abstol = 1.0f-8, maxiters = 500)
@test sol1.errors[:l2] < 0.5

# QuadratureTraining (Flux Chain)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))

(u_, t_) = (u_analytical(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(chain, opt, additional_loss = additional_loss)

sol1 = solve(prob, alg1, verbose = true, abstol = 1.0f-10, maxiters = 200)
@test sol1.errors[:l2] < 0.5

# QuadratureTraining (Lux Chain)
luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

(u_, t_) = (u_analytical(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(luxchain, opt, additional_loss = additional_loss)

sol1 = solve(prob, alg1, verbose = true, abstol = 1.0f-10, maxiters = 200)
@test sol1.errors[:l2] < 0.5

# StochasticTraining(Flux Chain)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))

(u_, t_) = (u_analytical(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(chain, opt, strategy = StochasticTraining(1000),
                       additional_loss = additional_loss)

sol1 = solve(prob, alg1, verbose = true, abstol = 1.0f-8, maxiters = 500)
@test sol1.errors[:l2] < 0.5

# StochasticTraining (Lux Chain)
luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

(u_, t_) = (u_analytical(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(luxchain, opt, strategy = StochasticTraining(1000),
                       additional_loss = additional_loss)

sol1 = solve(prob, alg1, verbose = true, abstol = 1.0f-8, maxiters = 500)
@test sol1.errors[:l2] < 0.5
