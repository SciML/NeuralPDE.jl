using Lux, ComponentArrays, OptimizationOptimisers
using Test, NeuralPDE
using Optimization
using CUDA, QuasiMonteCarlo
using ComponentArrays
import ModelingToolkit: Interval, infimum, supremum

using Random
Random.seed!(100)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end
CUDA.allowscalar(false)
#const gpuones = cu(ones(1))

## ODE
println("ode")
@parameters θ
@variables u(..)
Dθ = Differential(θ)

# 1D ODE
eq = Dθ(u(θ)) ~ θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
                u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

# Initial and boundary conditions
bcs = [u(0.0) ~ 1.0f0]

# Space and time domains
domains = [θ ∈ Interval(0.0f0, 1.0f0)]
# Discretization
dt = 0.1f0
# Neural network
inner = 20
chain = Chain(Dense(1, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, 1))

strategy = NeuralPDE.GridTraining(dt)
ps = Lux.setup(Random.default_rng(), chain)[1] |> ComponentArray |> gpu .|> Float64
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = ps)

@named pde_system = PDESystem(eq, bcs, domains, [θ], [u(θ)])
prob = NeuralPDE.discretize(pde_system, discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(1e-2); maxiters = 2000)
phi = discretization.phi

analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
ts = [infimum(d.domain):(dt / 10):supremum(d.domain) for d in domains][1]
u_real = [analytic_sol_func(t) for t in ts]
u_predict = [first(Array(phi([t], res.minimizer))) for t in ts]

@test u_predict≈u_real atol=0.2

# t_plot = collect(ts)
# plot(t_plot ,u_real)
# plot!(t_plot ,u_predict)

## 1D PDE Dirichlet boundary conditions
println("1D PDE Dirichlet boundary conditions")
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
    u(t, 0) ~ exp(-t),
    u(t, 1) ~ exp(-t) * cos(1)]

domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

inner = 30
chain = Lux.Chain(Dense(2, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, 1))

strategy = NeuralPDE.StochasticTraining(500)
ps = Lux.setup(Random.default_rng(), chain)[1] |> ComponentArray |> gpu .|> Float64
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = ps)
prob = NeuralPDE.discretize(pdesys, discretization)
symprob = NeuralPDE.symbolic_discretize(pdesys, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); maxiters = 1000)
phi = discretization.phi

u_exact = (t, x) -> exp.(-t) * cos.(x)
ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_predict = reshape([first(Array(phi([t, x], res.minimizer))) for t in ts for x in xs],
                    (length(ts), length(xs)))
u_real = reshape([u_exact(t, x) for t in ts for x in xs], (length(ts), length(xs)))
diff_u = abs.(u_predict .- u_real)

@test u_predict≈u_real atol=1.0

# p1 = plot(ts, xs, u_real, linetype=:contourf,title = "analytic");
# p2 = plot(ts, xs, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(ts, xs, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)

## 1D PDE Neumann boundary conditions and Float64 accuracy
println("1D PDE Neumann boundary conditions and Float64 accuracy")
@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
    Dx(u(t, 0)) ~ 0.0,
    Dx(u(t, 1)) ~ -exp(-t) * sin(1.0)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

inner = 20
chain = Lux.Chain(Dense(2, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, 1))

strategy = NeuralPDE.QuasiRandomTraining(500; #points
                                         sampling_alg = SobolSample(),
                                         resampling = false,
                                         minibatch = 30)
ps = Lux.setup(Random.default_rng(), chain)[1] |> ComponentArray |> gpu .|> Float64
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = ps)
prob = NeuralPDE.discretize(pdesys, discretization)
symprob = NeuralPDE.symbolic_discretize(pdesys, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.1); maxiters = 2000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 2000)
phi = discretization.phi

u_exact = (t, x) -> exp(-t) * cos(x)
ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_predict = reshape([first(Array(phi([t, x], res.minimizer))) for t in ts for x in xs],
                    (length(ts), length(xs)))
u_real = reshape([u_exact(t, x) for t in ts for x in xs], (length(ts), length(xs)))
diff_u = abs.(u_predict .- u_real)

@test u_predict≈u_real atol=1.0

# p1 = plot(ts, xs, u_real, linetype=:contourf,title = "analytic");
# p2 = plot(ts, xs, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(ts, xs, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)

## 2D PDE
println("2D PDE")
@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min = 0.0
t_max = 2.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0

# 3D PDE
eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
# Initial and boundary conditions
bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
    u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
    u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
    u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
    u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)]

# Neural network
inner = 25
chain = Lux.Chain(Dense(3, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, inner, Lux.σ),
                  Dense(inner, 1))

strategy = NeuralPDE.GridTraining(0.05)
ps = Lux.setup(Random.default_rng(), chain)[1] |> ComponentArray |> gpu .|> Float64
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = ps)

@named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
prob = NeuralPDE.discretize(pde_system, discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 2500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); maxiters = 2500)
@show res.original

phi = discretization.phi
ts, xs, ys = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
u_real = [analytic_sol_func(t, x, y) for t in ts for x in xs for y in ys]
u_predict = [first(Array(phi([t, x, y], res.minimizer))) for t in ts for x in xs
             for y in ys]

@test u_predict≈u_real rtol=0.2

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
