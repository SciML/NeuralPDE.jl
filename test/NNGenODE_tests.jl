using Test, Flux, NeuralNetDiffEq, Optim 
using DiffEqDevTools
using Random
using DiffEqFlux

Random.seed!(100)

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


# function lotka_volterra(du,u,p,t)
#   x, y = u
#   α, β, δ, γ = p
#   du[1] = dx = α*x - β*x*y
#   du[2] = dy = -δ*y + γ*x*y
# end
# lotka_volterra2 = (u,p,t) -> [p[1]*u[1] - p[2]*u[1]*u[2], -p[3]*u[2] + p[4]*u[1]*u[2]]
#
# u0 = [1.0f0,1.0f0]
# tspan = (0.0f0,4.0f0)
# p = [1.5f0,1.0f0,3.0f0,1.0f0]
# prob = ODEProblem(lotka_volterra,u0,tspan,p)
# sol_real = solve(prob,Tsit5())
#
# init_cond = [sol_real.u[1], sol_real.u[end]]
# chain = FastChain(FastDense(1,8,σ),FastDense(8,2))
# opt = Flux.ADAM(0.1)
# opt = BFGS()
# prob2 = ODEProblem(lotka_volterra2,u0,tspan,p)
# sol_pred  = solve(prob2,NeuralNetDiffEq.NNGenODE(chain,init_cond,opt),
#                   verbose = true, dt=1/5f0, maxiters=1000)
#
# Plots.plot(sol_real)
# Plots.plot!(sol_pred)
