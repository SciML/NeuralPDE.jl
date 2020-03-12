using Test, Flux
using DiffEqDevTools
using DiffEqBase, NeuralNetDiffEq

# Homogeneous on scalar
f = (du, u, p, t) -> -du+6*u
tspan = (0.0f0, 1.0f0)
u0 = 1.0f0
du0 = 0.0f0
dt = 1/5f0
f_analytic = t -> [1/5 * exp(-3*t) *(3*exp(5*t)+2.0)]
prob = SecondOrderODEProblem(ODEFunction{false}(f, analytic=f_analytic), u0, du0, tspan)
chain = Chain(Dense(1,5,σ),Dense(5,1))
opt = ADAM(1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=dt, verbose = true, maxiters = 100)
err1 = sqrt(sum((sol.u - [f_analytic(t)[1] for t in tspan[1]:dt:tspan[2]]).^2)[1])
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=1/20f0, verbose = true, maxiters = 100)
err2 = sqrt(sum((sol.u - [f_analytic(t)[1] for t in tspan[1]:1/20f0:tspan[2]]).^2)[1])
@test err2/err1 < 0.5


#Example 2: Test on vector
f = (du, u, p, t) -> -2*du-5*u+5*t^2+12
tspan = (0.0f0, 1.0f0)
u0 = [1.0f0]
du0 = [0.0f0]
dt = 1/5f0
f_analytic = t -> [1/50 * exp(-t) * (2*exp(t) * (25*t^2 - 20*t + 58) - 13*sin(2*t) - 66*cos(2*t))]
prob = SecondOrderODEProblem(ODEFunction{false}(f, analytic=f_analytic), u0, du0, tspan)
chain = Chain(Dense(1,5,σ),Dense(5,1))
opt = ADAM(1e-04, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=dt, verbose = true, maxiters = 100)
err1 = sqrt(sum((sol.u - [f_analytic(t)[1] for t in tspan[1]:dt:tspan[2]]).^2)[1])
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=1/20f0, verbose = true, maxiters = 100)
err2 = sqrt(sum((sol.u - [f_analytic(t)[1] for t in tspan[1]:1/20f0:tspan[2]]).^2)[1])
@test err2/err1 < 0.5
