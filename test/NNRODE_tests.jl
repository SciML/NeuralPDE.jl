println("RODE Tests")
using Flux , StochasticDiffEq, DiffEqNoiseProcess , Optim, Test
println("RODE_tests")
using  NeuralPDE

println("Test Case 1")
linear = (u,p,t,W) ->   2u*sin(W)
tspan = (0.00f0, 1.00f0)
u0 = 1.0f0
dt = 1/50f0
W = WienerProcess(0.0,0.0,nothing)
prob = RODEProblem(linear, u0 ,tspan, noise = W)
chain = Flux.Chain(Dense(2,8,relu),Dense(8,16 , relu),Dense(16,1))
opt = ADAM(1e-4)
sol = solve(prob, NeuralPDE.NNRODE(chain,W,opt), dt=dt, verbose = true,
            abstol=1e-10, maxiters = 5000)
W2 = NoiseWrapper(sol.W)
prob1 = RODEProblem(linear , u0 , tspan , noise = W2)
sol2 = solve(prob1,RandomEM(),dt=dt)
err = Flux.mse(sol.u , sol2.u)
@test err < 0.3

println("Test Case 2")
linear = (u,p,t,W) -> t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3))) + 5*W
tspan = (0.00f0, 1.00f0)
u0 = 1.0f0
dt = 1/100f0
W = WienerProcess(0.0,0.0,nothing)
prob = RODEProblem(linear, u0 ,tspan, noise = W)
chain = Flux.Chain(Dense(2,32,sigmoid),  Dense(32,32,sigmoid)  , Dense(32,1))
opt = ADAM(1e-3)
sol = solve(prob, NeuralPDE.NNRODE(chain,W,opt), dt=dt, verbose = true,
            abstol=1e-10, maxiters = 3500)
W2 = NoiseWrapper(sol.W)
prob1 = RODEProblem(linear , u0 , tspan , noise = W2)
sol2 = solve(prob1,RandomEM(),dt=dt)
err = Flux.mse(sol.u , sol2.u)
@test err < 0.4
