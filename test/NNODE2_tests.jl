using Test, Flux
using Plots
using DiffEqDevTools
using DiffEqBase, NeuralNetDiffEq

# Homogeneous
f(du, u, p, t) = -du+6*u
tspan = (0.0f0, 2.0f0)
u0 = [1.0f0]
du0 = [0.0f0]
dt = 1/20f0
function an_sol(du0, u0, p, t)
  [1/5 * exp(-3*t) *(3*exp(5*t)+2.0)]
end
prob = SecondOrderODEProblem(ODEFunction{false}(f, analytic=an_sol), u0, du0, tspan)
chain = Chain(Dense(1,5,σ),Dense(5,1))
opt = ADAM(1e-04, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=dt, verbose = true, abstol=1e-10, maxiters = 200)
println(sol)


#Example 2
f(du, u, p, t) = -2*du-5*u+5*t^2+12
tspan = (0.0f0, 2.0f0)
u0 = [1.0f0]
du0 = [0.0f0]
dt = 1/20f0
function an_sol(du0, u0, p, t)
  [1/50 * exp(-t) * (2*exp(t) * (25*t^2 - 20*t + 58) - 13*sin(2*t) - 66*cos(2*t))]
end
prob = SecondOrderODEProblem(ODEFunction{false}(f, analytic=an_sol), u0, du0, tspan)
chain = Chain(Dense(1,5,σ),Dense(5,1))
opt = ADAM(1e-04, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=dt, verbose = true, abstol=1e-10, maxiters = 200)
println(sol)


#= Much better accuracy:
#chain = Chain(
  x -> reshape(x, length(x), 1, 1), 
  MaxPool((1,)), 
  Conv((1,), 1=>16, relu),
  Conv((1,), 16=>32, relu), 
  Conv((1,), 32=>64, relu), 
  Conv((1,), 64=>256, relu), 
  Conv((1,), 256=>1028, relu), 
  Conv((1,), 1028=>1028), 
  x -> reshape(x, :, size(x, 4)), 
  Dense(1028, 512, tanh), 
  Dense(512, 128, relu), 
  Dense(128, 64, tanh), 
  Dense(64, 1))
=#
