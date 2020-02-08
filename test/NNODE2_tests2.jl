#using Plots
using Test, Flux
using DiffEqDevTools
using DiffEqBase, NeuralNetDiffEq

function f(ddu, du, u, p, t) #maybe turn to arrays too
	ddu[1] = (-p[3]*u[1]-p[4]*(u[1]-u[2]))/p[1]
	ddu[2] = (-p[4]*(u[2]-u[1]))/p[2]
end
function f(du, u, p, t)
	[(-p[3]*u[1]-p[4]*(u[1]-u[2]))/p[1], (-p[4]*(u[2]-u[1]))/p[2]]
end
p = [1.0f0, 1.0f0, 6.0f0, 4.0f0]
u0 = [1.0f0, 0.0f0]
du0 = [2.0f0, 0.0f0]
prob = SecondOrderODEProblem{false}(f, [u0], [du0], (0.0f0, 3.0f0), p)
dt = 1/20f0
opt = ADAM(1e-03)
chain = Chain(Dense(1,5,σ),Dense(5,2))
#=
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
=#
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain, opt), dt=dt, verbose=true, abstol=1e-10, maxiters=50)
#plot(sol)
