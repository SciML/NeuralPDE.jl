using Flux, OptimizationOptimisers, StochasticDiffEq, DiffEqNoiseProcess, Optim, Test
import Lux, OptimizationOptimJL
using NeuralPDE


using Random
Random.seed!(100)

println("Test Case 1")
linear = (u, p, t, W) -> 2u * sin(W)
tspan = (0.00f0, 1.00f0)
u0 = 1.0f0
dt = 1 / 50.0f0
W = WienerProcess(0.0, 0.0, nothing)
prob = RODEProblem(linear, u0, tspan, noise = W)
opt = OptimizationOptimisers.Adam(0.01)

W_test = solve(NoiseProblem(W, tspan), dt = dt)
prob1 = RODEProblem(linear, u0, tspan, noise = W_test)
analytical_sol = solve(prob1, RandomEM(), dt = dt)

ts = tspan[1]:dt:tspan[2]

chain = Flux.Chain(Dense(2, 8, relu), Dense(8, 16, relu), Dense(16, 1))
sol1 = solve(prob, NeuralPDE.NNRODE(chain, W, opt), dt = dt, verbose = true,
            abstol = 1e-10, maxiters = 500)
err = Flux.mse(vec(sol1[2](collect(ts), W_test.u)), analytical_sol.u)
@test err < 0.3        

chain = Flux.Chain(Dense(2, 8, relu), Dense(8, 16, relu), Dense(16, 1))
sol2 = solve(prob, NeuralPDE.NNRODE(chain, W, opt, batch = true), dt = dt, verbose = true, 
            abstol = 1e-10, maxiters = 500)
err = Flux.mse(vec(sol2[2](collect(ts), W_test.u)), analytical_sol.u)
@test err < 0.3             
            

luxchain = Lux.Chain(Lux.Dense(2, 8, relu), Lux.Dense(8, 16, relu), Lux.Dense(16, 1))
sol1 = solve(prob, NeuralPDE.NNRODE(luxchain, W, opt), dt = dt, verbose = true,
            abstol = 1e-10, maxiters = 500)
err = Flux.mse(vec(sol1[2](collect(ts), W_test.u)), analytical_sol.u)
@test err < 0.3        

luxchain = Lux.Chain(Lux.Dense(2, 8, relu), Lux.Dense(8, 16, relu), Lux.Dense(16, 1))
sol2 = solve(prob, NeuralPDE.NNRODE(luxchain, W, opt, batch = true), dt = dt, verbose = true, 
            abstol = 1e-10, maxiters = 500)
err = Flux.mse(vec(sol2[2](collect(ts), W_test.u)), analytical_sol.u)
@test err < 0.3             
            


println("Test Case 2")
linear = (u, p, t, W) -> t^3 + 2 * t + (t^2) * ((1 + 3 * (t^2)) / (1 + t + (t^3))) -
                         u * (t + ((1 + 3 * (t^2)) / (1 + t + t^3))) + 5 * W
tspan = (0.00f0, 1.00f0)
u0 = 1.0f0
dt = 1 / 100.0f0
W = WienerProcess(0.0, 0.0, nothing)
prob = RODEProblem(linear, u0, tspan, noise = W)
W_test = solve(NoiseProblem(W, tspan), dt = dt)
prob1 = RODEProblem(linear, u0, tspan, noise = W_test)
analytical_sol = solve(prob1, RandomEM(), dt = dt)

ts = tspan[1]:dt:tspan[2]

chain = Flux.Chain(Dense(2, 8, relu), Dense(8, 16, relu), Dense(16, 1))
sol1 = solve(prob, NeuralPDE.NNRODE(chain, W, opt), dt = dt, verbose = true,
            abstol = 1e-10, maxiters = 500)
err = Flux.mse(vec(sol1[2](collect(ts), W_test.u)), analytical_sol.u)
@test err < 0.3        

chain = Flux.Chain(Dense(2, 8, relu), Dense(8, 16, relu), Dense(16, 1))
sol2 = solve(prob, NeuralPDE.NNRODE(chain, W, opt, batch = true), dt = dt, verbose = true, 
            abstol = 1e-10, maxiters = 500)
err = Flux.mse(vec(sol2[2](collect(ts), W_test.u)), analytical_sol.u)
@test err < 0.3             

    luxchain = Lux.Chain(Lux.Dense(2, 8, relu), Lux.Dense(8, 16, relu), Lux.Dense(16, 1))
    sol1 = solve(prob, NeuralPDE.NNRODE(luxchain, W, opt), dt = dt, verbose = true,
                abstol = 1e-10, maxiters = 500)
    err = Flux.mse(vec(sol1[2](collect(ts), W_test.u)), analytical_sol.u)
    @test err < 0.3        

    luxchain = Lux.Chain(Lux.Dense(2, 8, relu), Lux.Dense(8, 16, relu), Lux.Dense(16, 1))
    sol2 = solve(prob, NeuralPDE.NNRODE(luxchain, W, opt, batch = true), dt = dt, verbose = true, 
                abstol = 1e-10, maxiters = 500)
    err = Flux.mse(vec(sol2[2](collect(ts), W_test.u)), analytical_sol.u)
    @test err < 0.3             
            
