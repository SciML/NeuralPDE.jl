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
prob = SecondOrderODEProblem(f, u0, du0, tspan)
chain = Chain(Dense(1,5,σ),Dense(5,1))
#chain = Chain(x -> reshape(x, length(x), 1, 1), MaxPool((1,)), Conv((1,), 1=>16, relu),Conv((1,), 16=>32, relu), Conv((1,), 32=>64, relu), Conv((1,), 64=>256, relu), Conv((1,), 256=>1028, relu), Conv((1,), 1028=>1028), x -> reshape(x, :, size(x, 4)), Dense(1028, 512, tanh), Dense(512, 128, relu), Dense(128, 64, tanh), Dense(64, 1))
opt = ADAM(0.1, (0.9, 0.95)) #0.05?
sol = solve(prob, NeuralNetDiffEq.NNODE2(chain,opt), dt=dt, verbose = true, abstol=1e-10, maxiters = 200)

t = tspan[1]:dt:tspan[2]
an_sol = @. 1/5 * exp(-3*t) *(3*exp(5*t)+2.0)

plot((t, an_sol), title="Second Order Linear ODE", label="Analytical", lw=3)
plot!((t, sol.u), label ="Numerical", lw=3)
xlabel!("t")
ylabel!("y(t)")
