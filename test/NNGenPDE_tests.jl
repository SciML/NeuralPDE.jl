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
linear_analytic(x,t) = (x - 3*t)*exp(-(x-3*t)^2)
boundary_cond_func(x) = x*exp(-x^2) #u(0,x)

opt = BFGS()
# opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,32,Flux.tanh),Dense(32,1))
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

boundary_conditions = boundary_cond_func.(xs)
initial_conditions = [linear_analytic.(xs[1],ts), linear_analytic.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u, phi ,res = NeuralNetDiffEq.solve(prob,alg,verbose = true, maxiters=20000)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)

##exmaple 2 t*du/dx + x du/dt = 0
tspan =(0.f0,0.5f0)
xspan = (0.f0,2.0f0)
dt= 0.1f0
dx =0.2f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

function pde_func(x,t,θ)
    t*dfdx(x,t,θ) + x*dfdt(x,t,θ)
end

linear_analytic(x,t) = x^2 +t^2
boundary_cond_func(x) = x^2 #u(0,x)
initial_condition_func(t) = t^2

# opt = BFGS()
opt = Flux.ADAM(0.01)
# chain = Chain(Dense(2,32,Flux.σ),Dense(32,1))
chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))

boundary_conditions = boundary_cond_func.(xs)
initial_conditions = [linear_analytic.(xs[1],ts) , linear_analytic.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u, phi ,res = NeuralNetDiffEq.solve(prob,alg,verbose = true, abstol = 1f-7, maxiters=10000)
u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)
dfdx = t -> (phi(t+sqrt(eps(typeof(dt)))) .- phi(t)) / sqrt(eps(typeof(dt)))
epsilon(t) = cbrt(eps(typeof(dt)))
#second order central
d2fdx2(t) = (phi(t+epsilon(t)) .- 2phi(t) .+ phi(t-epsilon(t)))/epsilon(t)^2
loss = () -> sum(abs2,sum(abs2,d2fdx2(t) .- f(dfdx(t), phi(t),p,t)) for t in ts) #[1] after f()
#example 3  5*du/dt + du/dx = x
tspan =(0.f0,1.0f0)
xspan = (0.f0,1.5f0)
dt= 0.2f0
dx =0.075f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

p = [5.0f0]
function pde_func(x,t,θ)
    5.0f0 * dfdt(x,t,θ) + dfdx(x,t,θ) - x
end

linear_analytic(x,t) = x^2/2 +sin(2pi*(5x -t)/5) - (5*x-t)^2/50
boundary_cond_func(x) = sin(2pi*x) #u(0,x)

# opt = BFGS()
opt = Flux.ADAM(0.01)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))

boundary_conditions = boundary_cond_func.(xs)
initial_conditions = [linear_analytic.(xs[1],ts) , linear_analytic.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, abstol=1f-20, maxiters=4000)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)

## example 4 Second order, Heat Equation du/dt = d2u/dx2
tspan =(0.f0,1.0f0)
xspan = (0.f0,2.0f0)
dt = 0.2f0
dx = 0.2f0
xs = xspan[1]:dx:xspan[end]
ts = tspan[1]:dt:tspan[end]

function pde_func(x,t,θ)
    dfdt(x,t,θ) - dfdxx(x,t,θ)
end

linear_analytic(x,t) =1/sqrt(1+4t)*exp(-x^2/(1+4t))
boundary_cond_func(x) = exp(-x^2) #u(0,x)

# opt = BFGS()
opt = Flux.ADAM(0.01)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(2,64,Flux.σ),FastDense(64,64,Flux.σ),FastDense(64,1))

boundary_conditions = boundary_cond_func.(xs)
initial_conditions = [linear_analytic.(xs[1],ts) , linear_analytic.(xs[end],ts)]

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, dt, dx)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=5000)

u_real = [[linear_analytic(x,t)  for x in xs ] for t in ts ]

x_plot = [x for x in xs]
plot(x_plot ,u)
plot!(x_plot ,u_real)

##example 5 ode
# tspan =(0.f0,1.0f0)
# xspan = (0.f0,0.1f0)
# dt= 0.25f0
# dx =0.1f0
# xs = xspan[1]:dx:xspan[end]
# ts = tspan[1]:dt:tspan[end]
#
# function pde_func(u,x,t,θ)
#      dfdt(x,t,θ) - cos(2pi*t)
# end
#
# boundary_cond_func(x) = -x * (x-1.f0) * sin(x)
#
# # opt = BFGS()
# opt = Flux.ADAM(0.01)
# # chain = Chain(Dense(2,32,Flux.σ),Dense(32,1))
# chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,1))
#
# boundary_conditions = boundary_cond_func.(xs)
# initial_conditions = [fill(0.1f0, length(ts)), fill(0.0f0, length(ts))] #u(t,0) ~ 0, u(t,1) ~ 0
#
# prob_NN = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,
#                                               boundary_conditions,
#                                               initial_conditions,
#                                               tspan, xspan, dt, dx)
# alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
# u_nn,phi,res  = NeuralNetDiffEq.solve(prob_NN,alg,verbose = true,abstol=1f-20, maxiters=500)
# x_plot = [x for x in xs]
# plot(x_plot ,u)
# plot!(x_plot ,u_real)
#
# # function pde_func(u,x,t,θ)
# #      dfdt(x,t,θ) - cos(2pi*t)
# # end
# # linear = (u,p,t) ->
# # linear_analytic = (u0,p,t) ->  exp(-t/5)*(u0 + sin(t))

##example 6
# tspan =(0.f0,1.0f0)
# xspan = (0.f0,1.0f0)
# dt= 0.25f0
# dx =0.1f0
# xs = xspan[1]:dx:xspan[end]
# ts = tspan[1]:dt:tspan[end]
#
# function pde_func(x,t,θ)
#     dfdt(x,t,θ) - dfdxx(x,t,θ)
# end
#
# boundary_cond_func(x) = -x * (x-1.f0) * sin(x)
#
# # opt = BFGS()
# opt = Flux.ADAM(0.01)
# # chain = Chain(Dense(2,32,Flux.σ),Dense(32,1))
# chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,1))
#
# boundary_conditions = boundary_cond_func.(xs)
# initial_conditions = [fill(0.1f0, length(ts)), fill(0.0f0, length(ts))] #u(t,0) ~ 0, u(t,1) ~ 0
#
# prob_NN = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,
#                                               boundary_conditions,
#                                               initial_conditions,
#                                               tspan, xspan, dt, dx)
# alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
# u_nn,phi,res  = NeuralNetDiffEq.solve(prob_NN,alg,verbose = true,abstol=1f-20, maxiters=500)
# x_plot = [x for x in xs]
# plot(x_plot, u_nn[1])
# plot!(x_plot, u_nn[2])
# plot!(x_plot, u_nn[3])
# plot!(x_plot, u_nn[4])
#
# using ModelingToolkit, DiffEqOperators, DiffEqBase, LinearAlgebra
# # Define some variables
# @parameters t x
# @variables u(..)
# @derivatives Dt'~t
# @derivatives Dxx''~x
# eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
# bcs = [u(0,x) ~ - x * (x-1) * sin(x),
#            u(t,0) ~ 0, u(t,1) ~ 0]
#
# domains = [t ∈ IntervalDomain(0.0,1.0),
#            x ∈ IntervalDomain(0.0,1.0)]
#
# pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
# discretization = MOLFiniteDifference(0.1)
# prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
#
# using OrdinaryDiffEq
# sol = solve(prob,Tsit5(),saveat=0.25)
#
# plot!(prob.space,Array(prob.extrapolation*sol[1]))
# plot!(prob.space,Array(prob.extrapolation*sol[2]))
# plot!(prob.space,Array(prob.extrapolation*sol[3]))
# plot!(prob.space,Array(prob.extrapolation*sol[4]))
