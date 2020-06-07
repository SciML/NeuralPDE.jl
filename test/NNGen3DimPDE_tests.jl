using Test, Flux,Optim#, NeuralNetDiffEq
using DiffEqDevTools
using Random
using DiffEqFlux
using Plots
using Adapt

dfdt = (x,y,t,θ) -> (phi(x,y,t+cbrt(eps(t)),θ) - phi(x,y,t,θ))/cbrt(eps(t))
dfdx = (x,y,t,θ) -> (phi(x+cbrt(eps(x)),y,t,θ) - phi(x,y,t,θ))/cbrt(eps(x))
dfdy = (x,y,t,θ) -> (phi(x,y+cbrt(eps(y)),t,θ) - phi(x,y,t,θ))/cbrt(eps(y))
epsilon(dv) = cbrt(eps(typeof(dv)))
dfdtt = (x,y,t,θ) -> (phi(x,y,t+epsilon(dt),θ) - 2phi(x,y,t,θ) + phi(x,y,t-epsilon(dt),θ))/epsilon(dt)^2
dfdxx = (x,y,t,θ) -> (phi(x+epsilon(dx),y,t,θ) - 2phi(x,y,t,θ) + phi(x-epsilon(dx),y,t,θ))/epsilon(dx)^2
dfdyy = (x,y,t,θ) -> (phi(x,y+epsilon(dy),t,θ) - 2phi(x,y,t,θ) + phi(x,y-epsilon(dy),t,θ))/epsilon(dy)^2

chain = FastChain(FastDense(3,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))
# initθ,re  = Flux.destructure(chain)
# phi = (x,t,θ) -> first(re(θ)(adapt(typeof(θ),collect([x;t]))))
initθ = DiffEqFlux.initial_params(chain)
phi = (x,y,t,θ) -> first(chain(adapt(typeof(θ),collect([x;y;t])),θ))

# Example 1   du2/dx2 + du2/dy2 =  du/dt
tspan =(0.f0,1.0f0)
xspan = (0.f0,2.0f0)
yspan = (0.0f0,2.0f0)
dt = 0.25f0
dx = 0.2f0
dy = 0.2f0
ts = tspan[1]:dt:tspan[end]
xs = xspan[1]:dx:xspan[end]
ys = yspan[1]:dy:yspan[end]

function pde_func(x,y,t,θ)
    dfdt(x,y,t,θ) - dfdxx(x,y,t,θ) - dfdyy(x,y,t,θ)
end

linear_analytic_func(x,y,t) = exp(x+y)*cos(x+y+4t)

cond_func(xs,ys,ts) =[linear_analytic_func(x,y,t) for x in xs for y in ys for t in ts]

# opt = BFGS()
opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

dom_ts = ts[2:end-1]
dom_xs = xs[2:end-1]
dom_ys = xs[2:end-1]

boundary_conditions = [cond_func(xs,ys, ts[1]); cond_func(xs,ys,ts[end]);
                       cond_func(xs[1],ys,dom_ts); cond_func(xs[end],ys,dom_ts);
                       cond_func(dom_xs,ys[1],dom_ts);cond_func(dom_xs,ys[end],dom_ts)]

initial_conditions = []

prob = NeuralNetDiffEq.GeneranNNTwoDimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, yspan, dt, dx, dy)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u_predict,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1500)

u_real = [reshape([linear_analytic_func(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]

p1 =plot(xs, ys, u_predict[2], st=:surface);
p2 = plot(xs, ys, u_real[2], st=:surface);
plot(p1,p2)

# Example 2  3dim Poisson equation du2/dx2 + du2/dy2+ du2/dt2 = -sin(pi*x)*sin(pi*y)*sin(pi*t)
tspan = (0.f0,1.0f0)
xspan = (0.f0,1.0f0)
yspan = (0.0f0,1.0f0)
dt = 0.2f0
dx = 0.1f0
dy = 0.1f0
ts = tspan[1]:dt:tspan[end]
xs = xspan[1]:dx:xspan[end]
ys = yspan[1]:dy:yspan[end]

function pde_func(x,y,t,θ)
    dfdtt(x,y,t,θ) + dfdxx(x,y,t,θ) + dfdyy(x,y,t,θ) + sin(pi*x)*sin(pi*y)*sin(pi*t)
end

linear_analytic_func(x,y,t) = sin(pi*x)*sin(pi*y)*sin(pi*t)/3pi^2

cond_func(xs,ys,ts) =[linear_analytic_func(x,y,t) for x in xs for y in ys for t in ts]

# opt = BFGS()
opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

dom_ts = ts[2:end-1]
dom_xs = xs[2:end-1]
dom_ys = xs[2:end-1]

boundary_conditions = [cond_func(xs,ys, ts[1]); cond_func(xs,ys,ts[end]);
                       cond_func(xs[1],ys,dom_ts); cond_func(xs[end],ys,dom_ts);
                       cond_func(dom_xs,ys[1],dom_ts);cond_func(dom_xs,ys[end],dom_ts)]

initial_conditions = []

prob = NeuralNetDiffEq.GeneranNN3DimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, yspan, dt, dx, dy)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u_predict,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1000)

u_real = [reshape([linear_analytic_func(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]

p1 =plot(xs, ys, u_predict[5], st=:surface);
p2 = plot(xs, ys, u_real[5], st=:surface);
plot(p1,p2)
