using Revise
using OrdinaryDiffEq, Plots, Flux
using NeuralNetDiffEq

function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end
function f(u,p,t)
  [p[1]*u[1] - p[2]*u[1]*u[2],-p[3]*u[2] + p[4]*u[1]*u[2]]
end

p = Float32[1.5,1.0,3.0,1.0]
u0 = Float32[1.0,1.0]
prob = ODEProblem(f,u0,(0f0,3f0),p)
prob_oop = ODEProblem{false}(f,u0,(0f0,3f0),p)

true_sol = solve(prob,Tsit5())

opt = ADAM(1e-03)


chain = Chain(
    x -> reshape(x, length(x), 1, 1),
    MaxPool((1,)),
    Conv((1,), 1=>16, relu),
    Conv((1,), 16=>16, relu),
    Conv((1,), 16=>32, relu),
    Conv((1,), 32=>64, relu),
    Conv((1,), 64=>256, relu),
    Conv((1,), 256=>256, relu),
    Conv((1,), 256=>1028, relu),
    Conv((1,), 1028=>1028),
    x -> reshape(x, :, size(x, 4)),
    Dense(1028, 512, tanh),
    Dense(512, 128, relu),
    Dense(128, 64, tanh),
    Dense(64, 2),
    softmax)

sol, losses, kicks, multipliers  = solve(prob_oop, NeuralNetDiffEq.NNODE(chain,opt) ,maxiters = 150, verbose = true, dt=1/5f0);

x = LinRange(1,size(losses)[1], size(losses)[1])
plot(x, losses, label = "total loss")
plot!(x,kicks, label = "Size of kick")
plot!(x, multipliers, label="Multiplier of kick")


plot(true_sol)
plot!(sol)
