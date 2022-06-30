using Test, Flux, Optim
using Random, NeuralPDE
Random.seed!(100)

# Run a solve on scalars
linear = (u, p, t) -> cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true,
            abstol = 1.0f-6, maxiters = 200)

opt = BFGS()
sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true,
            abstol = 1.0f-6, maxiters = 200)

# Run a solve on vectors
linear = (u, p, t) -> [cos(2pi * t)]
tspan = (0.0f0, 1.0f0)
u0 = [0.0f0]
prob = ODEProblem(linear, u0, tspan)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
opt = BFGS()
sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, abstol = 1e-10,
            verbose = true, maxiters = 200)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), abstol = 1.0f-6,
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
opt = ADAM(0.01)

sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

@test_throws Any solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true, maxiters = 400)

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = false, strategy = StochasticTraining(100)), verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = true, strategy = StochasticTraining(100)), verbose = true, maxiters = 400)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = false), verbose = true, maxiters = 400, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true, maxiters = 400, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

#Example 2
linear = (u, p, t) -> -u / 5 + exp(-t / 5) .* cos(t)
linear_analytic = (u0, p, t) -> exp(-t / 5) * (u0 + sin(t))
prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0f0, (0.0f0, 1.0f0))
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
opt = ADAM(0.1)
sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

@test_throws Any solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true, maxiters = 400,
                       abstol = 1.0f-8)

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = false, strategy = StochasticTraining(100)), verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = true, strategy = StochasticTraining(100)), verbose = true, maxiters = 400,
            abstol = 1.0f-8)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = false), verbose = true, maxiters = 400,
            abstol = 1.0f-8, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5

sol = solve(prob, NeuralPDE.NNODE(chain, opt; batch = true), verbose = true, maxiters = 400,
            abstol = 1.0f-8, dt = 1 / 5.0f0)
@test sol.errors[:l2] < 0.5
