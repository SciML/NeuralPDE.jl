using  NeuralNetDiffEq, Flux , DifferentialEquations, Optim, Adapt
using Plots
# linear = (u,p,t,W) -> t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3))) + 5*W
linear = (u,p,t,W) ->   2u*sin(W)
tspan = (0.00f0, 1.00f0)
u0 = 1.0f0
dt = 1/50f0
ts = tspan[1]:1/50f0:tspan[2]
t = ts

W = WienerProcess(0.0,0.0,nothing)
prob = RODEProblem(linear, u0 ,tspan, noise = W)
chain = Flux.Chain(Dense(2,512,elu),Dense(512,1))
opt = ADAM(0.0001)
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain,W,opt), dt=dt, verbose = true,
            abstol=1e-10, maxiters = 500)
W2 = NoiseWrapper(sol.W)
prob1 = RODEProblem(linear , u0 , tspan , noise = W2)
sol2 = solve(prob1,RandomEM(),dt=dt)
# real_sol = solve(prob , RandomEM() , dt = 1/20)
 plot(sol2,linewidth=5,title="Solution to the linear ODE with a thick line",
     xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!")

plot!(sol,linewidth=5,title="Solution to the linear ODE with a thick line",
          xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Sol!")
#
Flux.mse(sol2.u , sol.u)


# Flux.Optimiser(ExpDecay(1, 0.01, 9000, 1e-9), ADAM())
# noiseproblem = NoiseProblem(W2,(0.00 ,1.890))
# monte_prob = MonteCarloProblem(noiseproblem)
# W = solve(monte_prob;dt=dt,num_monte=100)
#
# μ = 1.0
# σ = 2.0
# prob = NoiseProblem(W,(0.0,1.0))
# monte_prob = MonteCarloProblem(prob)
# sol = solve(monte_prob;dt=0.1,num_monte=100)
#
# sol = solve(prob;dt=0.1)
#
# sol.W[]
