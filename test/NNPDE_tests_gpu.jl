using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using CUDA
using Quadrature
using QuasiMonteCarlo

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
CUDA.allowscalar(false)
#const gpuones = cu(ones(1))

## ODE
@parameters θ
@variables u(..)
Dθ = Differential(θ)

# 1D ODE
eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))

# Initial and boundary conditions
bcs = [u(0.) ~ 1.0]

# Space and time domains
domains = [θ ∈ IntervalDomain(0.0,1.0)]
# Discretization
dt = 0.1
# Neural network
inner = 20
chain = Chain(Dense(1,inner,Flux.σ),
              Dense(inner,inner,Flux.σ),
              Dense(inner,inner,Flux.σ),
              Dense(inner,inner,Flux.σ),
              Dense(inner,inner,Flux.σ),
              Dense(inner,1)) |> gpu

initθ = DiffEqFlux.initial_params(chain) |> gpu

strategy = NeuralPDE.GridTraining(dt)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = initθ
                                             )

pde_system = PDESystem(eq,bcs,domains,[θ],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, ADAM(1e-1); cb = cb, maxiters=1000)
phi = discretization.phi

analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
ts = [domain.domain.lower:dt/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(Array(phi([t],res.minimizer))) for t in ts]

@test u_predict ≈ u_real atol = 0.2

# t_plot = collect(ts)
# plot(t_plot ,u_real)
# plot!(t_plot ,u_predict)

## 1D PDE
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
bcs = [u(0,x) ~ cos(x),
        u(t,0) ~ exp(-t),
        u(t,1) ~ exp(-t) * cos(1)]

domains = [t ∈ IntervalDomain(0.0,1.0),
          x ∈ IntervalDomain(0.0,1.0)]

pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

inner = 30
chain = FastChain(FastDense(2,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,1))#,(u,p)->gpuones .* u)

strategy = NeuralPDE.StochasticTraining(500)
initθ = initial_params(chain) |>gpu
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = initθ)
prob = NeuralPDE.discretize(pdesys,discretization)
symprob = NeuralPDE.symbolic_discretize(pdesys,discretization)

res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,ADAM(0.001);cb=cb,maxiters=1000)
phi = discretization.phi

u_exact = (t,x) -> exp.(-t) * cos.(x)
ts,xs = [domain.domain.lower:0.01:domain.domain.upper for domain in domains]
u_predict = reshape([first(Array(phi([t,x],res.minimizer))) for t in ts for x in xs],(length(ts),length(xs)))
u_real = reshape([u_exact(t,x) for t in ts  for x in xs ], (length(ts),length(xs)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real atol = 1.0

# p1 = plot(ts, xs, u_real, linetype=:contourf,title = "analytic");
# p2 = plot(ts, xs, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(ts, xs, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)

## 2D PDE
@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.

# 3D PDE
eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
# Initial and boundary conditions
bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
       u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
       u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
       u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
       u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

# Space and time domains
domains = [t ∈ IntervalDomain(t_min,t_max),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)]

# Neural network
inner = 25
chain = FastChain(FastDense(3,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,1))#,(u,p)->gpuones .* u)

initθ = DiffEqFlux.initial_params(chain) |> gpu

# strategy = NeuralPDE.QuasiRandomTraining(3000; #points
#                                          sampling_alg = UniformSample(),
#                                          minibatch = 50)
# strategy = NeuralPDE.GridTraining(0.1)
strategy = NeuralPDE.StochasticTraining(4000)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = initθ)

pde_system = PDESystem(eq,bcs,domains,[t,x,y],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,ADAM(0.1);cb=cb,maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,ADAM(0.01);cb=cb,maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,ADAM(0.001);cb=cb,maxiters=1000)

phi = discretization.phi
ts,xs,ys = [domain.domain.lower:0.1:domain.domain.upper for domain in domains]
u_real = [analytic_sol_func(t,x,y) for t in ts for x in xs for y in ys]
u_predict = [first(Array(phi([t, x, y], res.minimizer))) for t in ts for x in xs for y in ys]

@test u_predict ≈ u_real atol = 20.0

# using Plots
# using Printf
#
# function plot_(res)
#     # Animate
#     anim = @animate for (i, t) in enumerate(0:0.05:t_max)
#         @info "Animating frame $i..."
#         u_real = reshape([analytic_sol_func(t,x,y) for x in xs for y in ys], (length(xs),length(ys)))
#         u_predict = reshape([Array(phi([t, x, y], res.minimizer))[1] for x in xs for y in ys], length(xs), length(ys))
#         u_error = abs.(u_predict .- u_real)
#         title = @sprintf("predict t = %.3f", t)
#         p1 = plot(xs, ys, u_predict,st=:surface, label="", title=title)
#         title = @sprintf("real")
#         p2 = plot(xs, ys, u_real,st=:surface, label="", title=title)
#         title = @sprintf("error")
#         p3 = plot(xs, ys, u_error, st=:contourf,label="", title=title)
#         plot(p1,p2,p3)
#     end
#     gif(anim,"3pde.gif", fps=10)
# end
#
# plot_(res)
