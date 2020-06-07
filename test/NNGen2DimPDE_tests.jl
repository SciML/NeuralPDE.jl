using Test, Flux,Optim#, NeuralNetDiffEq
using DiffEqDevTools
using Random
using DiffEqFlux

using Plots
using Adapt

dfdt = (x,t,θ) -> (phi(x,t+sqrt(eps(t)),θ) - phi(x,t,θ))/sqrt(eps(t))
dfdx = (x,t,θ) -> (phi(x+sqrt(eps(x)),t,θ) - phi(x,t,θ))/sqrt(eps(x))
# epsilon(val) = cbrt(eps(val))

epsilon(dv) = cbrt(eps(typeof(dv)))
#second order central
dfdtt = (x,t,θ) -> (phi(x,t+epsilon(dt),θ) - 2phi(x,t,θ) + phi(x,t-epsilon(dt),θ))/epsilon(dt)^2
dfdxx = (x,t,θ) -> (phi(x+epsilon(dx),t,θ) - 2phi(x,t,θ) + phi(x-epsilon(dx),t,θ))/epsilon(dx)^2

chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,1))
# initθ,re  = Flux.destructure(chain)
# phi = (x,t,θ) -> first(re(θ)(adapt(typeof(θ),collect([x;t]))))
initθ = DiffEqFlux.initial_params(chain)
phi = (x,t,θ) -> first(chain(adapt(typeof(θ),collect([x;t])),θ))

## Example 1  du/dt = -3du/dx
tspan =(0.f0,1.0f0)
xspan = (0.0f0,1.0f0)
dt= 0.2f0
dx =0.1f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

p =[3.0f0]
function pde_func(x,t,θ)
    dfdt(x,t,θ) + 3.0f0*dfdx(x,t,θ)
end
linear_analytic_func(x,t) = (x - 3*t)*exp(-(x-3*t)^2)

# opt = BFGS()
opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,32,Flux.tanh),Dense(32,1))
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

boundary_conditions =[linear_analytic_func.(xs,ts[1]),linear_analytic_func.(xs,ts[end])]
initial_conditions = [linear_analytic_func.(xs[1],ts), linear_analytic_func.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNN2DimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u, phi ,res = NeuralNetDiffEq.solve(prob,alg,verbose = true, maxiters=2000)

u_real = [[linear_analytic_func(x,t)  for x in xs ] for t in ts ]
u_predict = [[first(phi(x,t,res.minimizer)) for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u_predict)
plot!(x_plot ,u_real)

## Exmaple 2 t*du/dx + x du/dt = 0
tspan = (0.f0,10.0f0)
xspan = (0.f0,5.0f0)
dt= 2.5f0
dx =0.5f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

function pde_func(x,t,θ)
    t*dfdx(x,t,θ) + x*dfdt(x,t,θ)
end

linear_analytic_func(x,t) = x^2 +t^2

opt = Flux.ADAM(0.005)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

boundary_conditions = [linear_analytic_func.(xs,ts[1]),linear_analytic_func.(xs,ts[end])]
initial_conditions = [linear_analytic_func.(xs[1],ts), linear_analytic_func.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNN2DimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u, phi ,res = NeuralNetDiffEq.solve(prob,alg,verbose = true, abstol = 1f-7, maxiters=10000)

u_real = [[linear_analytic_func(x,t)  for x in xs ] for t in ts ]
u_predict = [[first(phi(x,t,res.minimizer)) for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u_real)
plot!(x_plot ,u_predict)

## Example 3  5*du/dt + du/dx = x
tspan =(0.f0,1.0f0)
xspan = (0.f0,3.0f0)
dt= 0.2f0
dx =0.05f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

p = [5.0f0]
function pde_func(x,t,θ)
    5.0f0 * dfdt(x,t,θ) + dfdx(x,t,θ) - x
end

linear_analytic_func(x,t) = x^2/2 +sin(2pi*(5x -t)/5) - (5*x-t)^2/50
boundary_cond_func(x) = sin(2pi*x) #u(0,x)

opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

boundary_conditions = [linear_analytic_func.(xs,ts[1]),linear_analytic_func.(xs,ts[end])]
initial_conditions = [linear_analytic_func.(xs[1],ts) , linear_analytic_func.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNN2DimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, abstol=1f-20, maxiters=2000)

u_real = [[linear_analytic_func(x,t)  for x in xs ] for t in ts ]
u_predict = [[first(phi(x,t,res.minimizer)) for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)

## Example 4 Second order, Heat Equation du/dt = d2u/dx2
tspan =(0.f0,1.0f0)
xspan = (0.f0,2.0f0)
dt = 0.2f0
dx = 0.2f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

function pde_func(x,t,θ)
    dfdt(x,t,θ) - dfdxx(x,t,θ)
end

linear_analytic_func(x,t) =1/sqrt(1+4t)*exp(-x^2/(1+4t))

opt = Flux.ADAM(0.01)
chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))

boundary_conditions = [linear_analytic_func.(xs,ts[1]),linear_analytic_func.(xs,ts[end])]
initial_conditions = [linear_analytic_func.(xs[1],ts), linear_analytic_func.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNN2DimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1500)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts]
u_predict = [[first(phi(x,t,res.minimizer)) for x in xs ] for t in ts]

x_plot = [x for x in xs]
plot(x_plot ,u_predict)
plot!(x_plot ,u_real)
#
u_real_surf =  reshape([linear_analytic(x,t) for x in xs for t in ts],(length(xs),length(ts)))
u_predict_surf = reshape([first(phi(x,t,res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
p1 =plot(xs, ts, u_real_serf, st=:surface);
p2 = plot(xs, ts, u_predict_serf, st=:surface);
plot(p1,p2)


## example 5 Poisson equation
tspan =(0.f0,1.0f0)
xspan = (0.f0,1.0f0)
dt= 0.1f0
dx =0.1f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

function pde_func(x,t,θ)
    dfdxx(x,t,θ) + dfdtt(x,t,θ) + 1
end

boundary_cond_func(x,t) =0.f0 #u(x,y) = 0 on edge

opt = Flux.ADAM(0.01)
chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))

boundary_conditions = [boundary_cond_func.(xs,ts[1]),boundary_cond_func.(xs,ts[end])]
initial_conditions = [boundary_cond_func.(xs[1],ts), boundary_cond_func.(xs[end],ts)]
prob = NeuralNetDiffEq.GeneranNN2DimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1000)

u_predict = reshape([first(phi(x,t,res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

plot(xs, ts, u_predict, st=:surface)
