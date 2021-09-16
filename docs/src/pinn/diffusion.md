##1D diffusion

```julia
using NeuralPDE
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba
using Plots, Printf

@parameters t x
@variables u(..), Dxu(..), O1(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# PDE and boundary conditions
eq  = Dt(u(t, x)) ~ Dx(Dxu(t, x))

bcs_ = [u(0, x) ~ cos(x),
       Dx(u(t, 0)) ~ 0,
       Dx(u(t, 1)) ~ exp(-t) * -sin(1)]

ep = (cbrt(eps(eltype(Float64))))^2 / 6

der = [Dxu(t, x) ~ Dx(u(t, x)) + ep * O1(t, x)]

bcs = [bcs_;der]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# Neural network
chain = [[FastChain(FastDense(2, 20, Flux.tanh), FastDense(20, 20, Flux.tanh), FastDense(20, 1)) for _ in 1:2];
         [FastChain(FastDense(2, 6, Flux.tanh), FastDense(6, 1)) for _ in 1:1];]

initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

strategy = NeuralPDE.GridTraining(0.05)
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy;init_params=initθ)
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u,Dxu,O1])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
bcs_inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents

cb_ = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = GalacticOptim.solve(prob, BFGS();cb=cb_,maxiters=1000,g_tol=10e-15)
prob =remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, BFGS();cb=cb_,maxiters=10000,g_tol=10e-15)

phi = discretization.phi[1]

# Analysis
ts, xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
analytic_sol_func(t, x) = exp.(-t) * cos.(x)

# Plot
using Plots
using Printf

u_predict = reshape([first(phi([t,x], res.minimizer)) for t in ts for x in xs], (length(ts), length(xs)))
u_real = reshape([analytic_sol_func(t, x) for t in ts for x in xs], (length(ts), length(xs)))

diff_u = abs.(u_predict .- u_real)
p1 = plot(ts, xs, u_real, linetype=:contourf, title="analytic");
p2 = plot(ts, xs, u_predict, linetype=:contourf, title="predict");
p3 = plot(ts, xs, diff_u, linetype=:contourf, title="error");
```



##Advection diffusion

```julia
using NeuralPDE
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba
using Plots, Printf

@parameters t,x,y
@variables c(..)
Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dx = Differential(x)
Dy = Differential(y)

@variables Dxc(..), Dyc(..), O1(..), O2(..)

# Parameters
u = 1.0
v = 1.0
v_vector = (u, v)
R = 0
D = 0.0 # diffusion
t_min = 0.0
t_max = 1.0
x_min = -1.0
x_max = 1.0
y_min = -1.0
y_max = 1.0
d = y_max-y_min

cb = function (p,l)
    println("Current loss is: $l")
    return false
end


# exp(-(x^2+y^2)/0.1)
# div(v_vector * c) = dx(uc) + dy(vc)
# Equations, initial and boundary conditions

eqs = Dt(c(t,x,y)) ~ D * (Dx(Dxc(t,x,y)) + Dy(Dyc(t,x,y))) - (u*Dx(c(t,x,y)) + v*Dy(c(t,x,y))) + R

bcs_ = [
        c(0, x, y) ~ exp(-(x^2+y^2)/0.1), #cos(π*x) * cos(π*y) + 1.0 #,
        c(t, x, y_min) ~ c(t, x, y_max),
        c(t, x_min, y) ~ c(t, x_max, y)
      ]

der = [Dxc(t, x, y) ~ Dx(c(t, x, y))+O1(t, x, y),
       Dyc(t, x, y) ~ Dy(c(t, x, y))+O2(t, x, y)]

bcs = [bcs_;der;]

domains = [t ∈ Interval(t_min,t_max),
           x ∈ Interval(x_min,x_max),
           y ∈ Interval(y_min,y_max)
          ]

dim = length(domains)
hidden = 20
hidden_der = 10
chain = [[FastChain(FastDense(3, hidden, Flux.tanh),
                    FastDense(hidden, hidden, Flux.tanh),
                    FastDense(hidden, 1)) for _ in 1:1];
         [FastChain(FastDense(3, hidden_der, Flux.tanh),
                    FastDense(hidden_der, hidden_der, Flux.tanh),
                    FastDense(hidden_der, 1)) for _ in 1:2];
         [FastChain(FastDense(3, 6, Flux.tanh), FastDense(6, 1)) for _ in 1:2];]

initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

grid_strategy = NeuralPDE.GridTraining(0.1)
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubaCuhre(),
                                                   reltol = 1e-4, abstol = 1e-2,
                                                   maxiters = 20, batch=100)

discretization = NeuralPDE.PhysicsInformedNN(chain,quadrature_strategy, init_param = initθ)
pde_system = PDESystem(eqs, bcs, domains, [t,x,y], [c, Dxc, Dyc, O1, O2])
@named  prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

flat_initθ = reduce(vcat, initθ)
prob.f.f.loss_function(reduce(vcat, flat_initθ))

prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,ADAM(0.1);cb=cb,maxiters=200)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=1000)

function plot_comparison(name)
    # Animate
    anim = @animate for (i, t) in enumerate(0:dt:t_max)
        @info "Animating frame $i..."
        title = @sprintf("Advection-diffusion t = %.3f", t)
        c_real = reshape([analytic_solve(t,x,y) for x in xs for y in ys], (length(xs),length(ys)))
        c_predict = reshape([phi([t, x, y], res.minimizer)[1] for x in xs for y in ys], length(xs), length(ys))
        error_c = abs.(c_predict .- c_real)
        title = @sprintf("predict")
        p1 = heatmap(xs, ys, c_predict, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal, clims=(0, 1.))
        title = @sprintf("real")
        p2 = heatmap(xs, ys, c_real, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal, clims=(0, 1.))
        title = @sprintf("error")
        p3 = heatmap(xs, ys, error_c, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal,clims=(0, 0.03))
        plot(p1,p2,p3)
    end
    gif(anim, name, fps=15)
end

nx = 64
ny = 64
dx = (x_max-x_min) / (nx - 1)
dy = (y_max-y_min) / (ny -1)
xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains[2:3]]

plot_comparison("advection_diffusion_2d_pinn_comparison_high.gif")
```

![advection_diffusion_2d_pinn_comparison](https://user-images.githubusercontent.com/12683885/129394043-1e11d3d7-0154-4101-a083-7b376fad4e73.gif)
