
using Test, Flux, NeuralNetDiffEq, DifferentialEquations
using DiffEqDevTools
#Example 1
function f(u,p,t,W)
  2u*sin(W)
end
u0 = 0.00
tspan = (0.0,5.0)
prob = RODEProblem(f,u0,tspan)

chain = Flux.Chain(Dense(2,5,σ),Dense(5,16 ,σ ) , Dense(16,2))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain,opt), dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 1000)

# Example 2
f2 = (u,p,t,W) -> 1.01u.+0.87u.*W
u0 = 1.00
tspan = (0.0,1.0)
prob = RODEProblem(f2,u0,tspan)
chain = Flux.Chain(Dense(2,5,σ),Dense(5,16 ,σ ) , Dense(16,2))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain,opt), dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 1000)






