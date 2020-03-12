#using Plots
using Test, Flux
using DiffEqDevTools
using DiffEqBase, NeuralNetDiffEq

function f(ddu, du, u, p, t)
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
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain, opt), dt=dt, verbose=true, abstol=1e-10, maxiters=50)
