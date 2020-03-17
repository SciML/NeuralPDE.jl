using Test, Flux,Optim#, NeuralNetDiffEq
using DiffEqDevTools
using Random
using DiffEqFlux
# using ModelingToolkit, DiffEqOperators, DiffEqBase, LinearAlgebra
using Plots

Random.seed!(100)

dfdt = (x,t,θ) -> (phi(x,t+sqrt(eps(t)),θ) - phi(x,t,θ))/sqrt(eps(t))
dfdx = (x,t,θ) -> (phi(x+sqrt(eps(x)),t,θ) - phi(x,t,θ))/sqrt(eps(x))
phi = (x,t,θ) -> first(chain([x;t],θ))

Example 1  du/dt = -3du/dx
tspan =(0.f0,2.0f0)
xspan = (0.0f0,2.0f0)
dt= 0.2f0
dx =0.1f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

p =[3.0f0]
function pde_func(x,t,θ)
    dfdt(x,t,θ) + 3.0f0*dfdx(x,t,θ)
end
linear_analytic(x,t) = (x - 3*t)*exp(-(x-3*t)^2)
boundary_cond_func(x) = x*exp(-x^2) #u(0,x)

opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,1))

u_0 = @. boundary_cond_func(xs)
un = @. linear_analytic(xs,ts[end])
boundary_conditions = [u_0, un]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u  = NeuralNetDiffEq.solve(prob,alg,verbose = true, maxiters=5000)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)
plot!(x_plot ,u_0)


# #exmaple 2 t*du/dx + x du/dt = 0
tspan =(0.f0,2.0f0)
xspan = (0.f0,2.0f0)
dt= 1.0f0
dx =0.1f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

function pde_func(x,t,θ)
    t*dfdx(x,t,θ) + x*dfdt(x,t,θ)
end

linear_analytic(x,t) = x^2 +t^2
boundary_cond_func(x) = x^2 #u(0,x)

opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(2,8,Flux.σ),FastDense(8,1))

u_0 = @. boundary_cond_func(xs)
un = @. linear_analytic(xs,ts[end])
boundary_conditions = [u_0, un]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u  = NeuralNetDiffEq.solve(prob,alg,verbose = true, maxiters=5000)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)


#example 3
tspan =(0.f0,1.0f0)
xspan = (0.f0,2.0f0)
dt= 0.5f0
dx =0.1f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]
p = [5.0f0]
function pde_func(x,t,θ)
    5.0f0 * dfdt(x,t,θ) + dfdx(x,t,θ) - x
end

linear_analytic(x,t) = x^2/2 +sin(2pi*(5x -t)/5) - (5*x-t)^2/50
boundary_cond_func(x) = sin(2pi*x) #u(0,x)


opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,1))

u_0 = @. boundary_cond_func(xs)
un = @. linear_analytic(xs,ts[end])
boundary_conditions = [u_0, un]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u  = NeuralNetDiffEq.solve(prob,alg,verbose = true, maxiters=5000)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)
