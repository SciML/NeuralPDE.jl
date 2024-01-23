using Test, Flux
using Random#, NeuralPDE
using OrdinaryDiffEq, Optimisers, Statistics
import Lux, OptimizationOptimisers, OptimizationOptimJL

Random.seed!(100)

#Example 1
function example1(du, u, p, t)
    du[1] = cos(2pi * t)
    du[2] = u[2] + cos(2pi * t)
    nothing
end
M = [1.0 0
    0 0]
f = ODEFunction(example1, mass_matrix = M)
tspan = (0.0f0, 1.0f0)
prob_mm = ODEProblem(f, [1.0, -1.0], tspan)
sol1 = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

#DAE
example = (du, u, p, t) -> [cos(2pi * t), u[2] + cos(2pi * t)]
u₀ = [1.0, -1.0]
du₀ = [0.0, 0.0]
tspan = (0.0f0, 1.0f0)
differential_vars = [true, false]
prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
# prob = ODEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
chain = Flux.Chain(Dense(1, 15, cos), Dense(15, 15, sin), Dense(15, 2))
opt = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, opt; autodiff = false)

sol = solve(prob,
    alg, verbose = true, dt = 1 / 100.0f0,
    maxiters = 3000, abstol = 1.0f-10)

plot(sol1, tspan = tspan, layout = (2, 1))
plot!(sol, tspan = tspan, layout = (2, 1))

@test sol.errors[:l2] < 0.5
