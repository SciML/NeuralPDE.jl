using  NeuralNetDiffEq, Flux , DifferentialEquations, Optim, Adapt
using Plots
# linear = (u,p,t,W) -> 10*cos(2*pi*t) + 5*W
linear = (u,p,t,W) ->   2u*sin(W)
tspan = (0.0f0, 5.0f0)
u0 = 1.0f0
ts = tspan[1]:1/20f0:tspan[2]
t = ts
dt = 1/20f0
brownian_values = cumsum([0;[sqrt(dt)*randn() for i in 1:length(t)-1]])
W = NoiseGrid(t,brownian_values)
prob = RODEProblem(linear, u0 ,tspan, noise = W)
chain = Flux.Chain(Dense(2,128,elu),Dense(128,1))
opt = ADAM(0.001)
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain,W,opt), dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 2500)
W2 = NoiseWrapper(W)
prob1 = RODEProblem(linear , u0 , tspan , noise = W2)
sol2 = solve(prob1,RandomEM(),dt=dt)
# real_sol = solve(prob , RandomEM() , dt = 1/20)
# plot(sol2,linewidth=5,title="Solution to the linear ODE with a thick line",
#      xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!")
#
# plot!(sol,linewidth=5,title="Solution to the linear ODE with a thick line",
#           xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Sol!")
#
# Flux.mse(sol2.u , sol.u)


# Flux.Optimiser(ExpDecay(1, 0.01, 9000, 1e-9), ADAM())
