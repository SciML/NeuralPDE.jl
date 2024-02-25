## 2D Poisson's equation

Let's solve the following 2-dimensional Poisson's equation:

```math
\begin{align*}
∂^2_x u(x, y) + ∂^2_y u(x, y) = -sin (\pi x) sin (\pi y) \quad & \textsf{for all } 0 < x, y < 1 \, , \\
u(0, y) = u(1, y) = 0 \quad & \textsf{for all } 0 < y < 1 \, , \\
u(x, 0) = u(x, 1) = 0 \quad & \textsf{for all } 0 < x < 1 \, , \\
\end{align*}
```

We obtain the solution of this equation with the given boundary conditions using Deep Galerkin Method:

```@example dgm_poisson
using NeuralPDE
using ModelingToolkit, Optimization, OptimizationOptimisers
import Lux: tanh, identity
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# Initial and boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
    u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

strategy = QuasiRandomTraining(4_000, minibatch= 500);
discretization= DeepGalerkin(2, 1, 30, 3, tanh, tanh, identity, strategy);

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

global iter = 0;
callback = function (p, l)
    global iter += 1;
    if iter%50 == 0
        println("$iter => $l")
    end
    return false
end

res = Optimization.solve(prob, ADAM(0.01); callback = callback, maxiters = 600)
phi = discretization.phi
```

We now plot the predicted solution of the PDE and compare it with the analytical solution to plot the relative error.

```@example dgm_poisson
using Plots

xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.minimizer)) for x in xs for y in ys],
                    (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                (length(xs), length(ys)))

diff_u = abs.(u_predict .- u_real)

using Plots
p1 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic");
p2 = plot(xs, ys, u_predict, linetype = :contourf, title = "predict");
p3 = plot(xs, ys, diff_u, linetype = :contourf, title = "error");
plot(p1, p2, p3)
```