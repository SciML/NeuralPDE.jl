using Test, Flux, NeuralNetDiffEq , StochasticDiffEq , LinearAlgebra
using DiffEqDevTools
d = 1
r = 0.04
delta = 0.0
beta = 0.2
T = 1
u0 = fill(90.00 , d , 1)
sdealg = EM()
ensemblealg = EnsembleThreads()
f(u,p,t) = (r-delta)*u
sigma(u,p,t)  = beta*u
tspan = (0.0 , 1.0)
N = 50
dt = tspan[2]/49
K = 100.00
function g(t , x)
  return exp(-r*t)*(max(K -  maximum(x)  , 0))
end

prob  = OptimalStoppingProblem(f , sigma  , g , u0 , tspan)
opt = Flux.ADAM(0.1)
m = Chain(Dense(d , 32, sigmoid), Dense(32, 32 , sigmoid)  , Dense(32 , N ), softmax)
sol = solve(prob, NeuralNetDiffEq.NNStopping( m,opt , sdealg , ensemblealg), verbose = true, dt = dt,
            abstol=1e-6, maxiters = 50 , trajectories = 100)
