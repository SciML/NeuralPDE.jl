using Test, Flux, Optim
using DiffEqDevTools
using Random
using DiffEqFlux

#Example 1
linear = (u,p,t) -> @. t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3)))
linear_analytic = (u0,p,t) -> [exp(-(t^2)/2)/(1+t+t^3) + t^2]
tspan = (0.0f0, 1.0f0)
init_cond = [linear_analytic(0.,0., tspan[1]), linear_analytic(0.,0., tspan[end])]
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),[1f0],(0.0f0,1.0f0))
chain = Flux.Chain(Dense(1,128,σ),Dense(128,1))
opt = ADAM(0.01)
sol  = solve(prob,NeuralNetDiffEq.NNGenODE(chain,init_cond,opt),verbose = true,
             dt=1/20f0, maxiters=200)
@test sol.errors[:l2] < 0.5


#Example 2
linear = (u,p,t) -> -u/5 + exp(-t/5).*cos(t)
linear_analytic = (u0,p,t) ->  exp(-t/5)*(u0 + sin(t))
tspan = (0.0f0, 1.0f0)
init_cond = [linear_analytic(0.,0., tspan[1]), linear_analytic(0.,0., tspan[end])]
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),0.0f0,(0.0f0,1.0f0))
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
# chain = FastChain(FastDense(1,5,σ),FastDense(5,1))
opt = ADAM(0.01)
sol  = solve(prob,NeuralNetDiffEq.NNGenODE(chain,init_cond,opt),verbose = true, dt=1/5f0)
@test sol.errors[:l2] < 0.5
