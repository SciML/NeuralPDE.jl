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

# Example 1  Poisson equation du2/dx2 + du2/dy2 =  x^2 + y^2
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
linear_analytic(x,y,t) = exp(x+y)*cos(x+y+4t)
boundary_cond_func(xs, ys) = [linear_analytic(x,y,0.0f0) for x in xs, y in ys]
initial_cond_func(xs,ys,ts) =[linear_analytic(x,y,t) for x in xs, y in ys, t in ts]
# opt = BFGS()
opt = Flux.ADAM(0.1)
# chain = Chain(Dense(2,16,Flux.σ),Dense(16,1))
chain = FastChain(FastDense(3,42,Flux.σ),FastDense(42,42,Flux.σ),FastDense(42,1))

boundary_conditions = boundary_cond_func(xs, ys)
dom_ts = ts[2:end]
dom_xs = xs[2:end-1]
initial_conditions = [initial_cond_func(xs[1],ys,dom_ts), initial_cond_func(xs[end],ys,dom_ts),
                      initial_cond_func(dom_xs,ys[1],dom_ts),initial_cond_func(dom_xs,ys[end],dom_ts)]

prob = NeuralNetDiffEq.GeneranNNTwoDimPDEProblem(pde_func,boundary_conditions,initial_conditions,tspan, xspan, yspan, dt, dx, dy)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
u_nn,phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=2000)

u_real = [reshape([linear_analytic(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]

p1 =plot(xs, ys, u_nn[end-1], st=:surface);
p2 = plot(xs, ys, u_real[end-1], st=:surface);
plot(p1,p2)
