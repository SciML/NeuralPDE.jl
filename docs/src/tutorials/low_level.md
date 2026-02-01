# Investigating `symbolic_discretize` with the `PhysicsInformedNN` Discretizer for the 1-D Burgers' Equation

Let's consider the Burgers' equation:

```math
\begin{gather*}
∂_t u + u ∂_x u - (0.01 / \pi) ∂_x^2 u = 0 \, , \quad x \in [-1, 1], t \in [0, 1] \, , \\
u(0, x) = - \sin(\pi x) \, , \\
u(t, -1) = u(t, 1) = 0 \, ,
\end{gather*}
```

with Physics-Informed Neural Networks. Here is an example of using the low-level API:

```@example low_level
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches
using DomainSets: Interval
using IntervalSets: leftendpoint, rightendpoint

@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

# Initial and boundary conditions
bcs = [u(0, x) ~ -sin(pi * x),
    u(t, -1) ~ 0.0,
    u(t, 1) ~ 0.0,
    u(t, -1) ~ u(t, 1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(-1.0, 1.0)]

# Neural network
chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
strategy = QuadratureTraining(; abstol = 1e-6, reltol = 1e-6, batch = 200)

indvars = [t, x]
depvars = [u(t, x)]
@named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

discretization = PhysicsInformedNN(chain, strategy)
sym_prob = symbolic_discretize(pde_system, discretization)

phi = sym_prob.phi

pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p.u), pde_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p.u), bc_loss_functions))
    return false
end

loss_functions = [pde_loss_functions; bc_loss_functions]

loss_function(θ, p) = sum(map(l -> l(θ), loss_functions))

f_ = OptimizationFunction(loss_function, AutoZygote())
prob = OptimizationProblem(f_, sym_prob.flat_init_params)

res = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 500)
```

And some analysis:

```@example low_level
using Plots

ts, xs = [leftendpoint(d.domain):0.01:rightendpoint(d.domain) for d in domains]
u_predict_contourf = reshape([first(phi([t, x], res.u)) for t in ts for x in xs],
    length(xs), length(ts))
plot(ts, xs, u_predict_contourf, linetype = :contourf, title = "predict")

u_predict = [[first(phi([t, x], res.u)) for x in xs] for t in ts]
p1 = plot(xs, u_predict[3], title = "t = 0.1");
p2 = plot(xs, u_predict[11], title = "t = 0.5");
p3 = plot(xs, u_predict[end], title = "t = 1");
plot(p1, p2, p3)
```
