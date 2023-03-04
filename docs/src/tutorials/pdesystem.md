# Specifying and Solving PDESystems with Physics-Informed Neural Networks (PINNs)

In this example, we will solve a Poisson equation:

```math
∂^2_x u(x, y) + ∂^2_y u(x, y) = - \sin(\pi x) \sin(\pi y) \, ,
```

with the boundary conditions:

```math
\begin{align*}
u(0, y) &= 0 \, ,\\
u(1, y) &= 0 \, ,\\
u(x, 0) &= 0 \, ,\\
u(x, 1) &= 0 \, ,
\end{align*}
```

on the space domain:

```math
x \in [0, 1] \, , \ y \in [0, 1] \, ,
```

with grid discretization `dx = 0.05` using physics-informed neural networks.

## Copy-Pasteable Code

```@example
using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# Boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]

# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx))

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

#Optimizer
opt = OptimizationOptimJL.BFGS()

#Callback function
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters = 1000)
phi = discretization.phi

using Plots

xs, ys = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
                    (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                 (length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic");
p2 = plot(xs, ys, u_predict, linetype = :contourf, title = "predict");
p3 = plot(xs, ys, diff_u, linetype = :contourf, title = "error");
plot(p1, p2, p3)
```

## Detailed Description

The ModelingToolkit PDE interface for this example looks like this:

```@example poisson
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x y
@variables u(..)
@derivatives Dxx'' ~ x
@derivatives Dyy'' ~ y

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# Boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]
```

Here, we define the neural network, where the input of NN equals the number of dimensions and output equals the number of equations in the system.

```@example poisson
# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))
```

Here, we build PhysicsInformedNN algorithm where `dx` is the step of discretization where
`strategy` stores information for choosing a training strategy.

```@example poisson
# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx))
```

As described in the API docs, we now need to define the `PDESystem` and create PINNs
problem using the `discretize` method.

```@example poisson
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)
```

Here, we define the callback function and the optimizer. And now we can solve the PDE using PINNs
(with the number of epochs `maxiters=1000`).

```@example poisson
#Optimizer
opt = OptimizationOptimJL.BFGS()

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters = 1000)
phi = discretization.phi
```

We can plot the predicted solution of the PDE and compare it with the analytical solution to plot the relative error.

```@example poisson
xs, ys = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
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

![poissonplot](https://user-images.githubusercontent.com/12683885/90962648-2db35980-e4ba-11ea-8e58-f4f07c77bcb9.png)
