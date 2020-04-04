using  NeuralNetDiffEq, Flux , DifferentialEquations, Optim, Adapt
f = (u,p,t,W) -> 1.01u.+0.87u.*W
u0 = 1.00
tspan = (0.0,1.0)
dt = 0.1/50f0
ts= tspan[1]:dt:tspan[2]
W = NoiseGrid(collect(ts) , randn(length(ts)))
prob = RODEProblem(f,u0,tspan , noise = W)

chain = Flux.Chain(Dense(2,128,elu),Dense(128,128,sigmoid),Dense(128,1))
opt = BFGS()
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain, W ,opt), dt=1/50f0,
verbose = true, abstol=1e-10, maxiters = 100)

sol_real = solve(prob,RandomEM(),dt=1/50)

using Plots
plot(sol_real,linewidth=5,title="Solution to the linear ODE with a thick line",
     xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!")
sol
Flux.mse(sol.u , sol_real.u)
