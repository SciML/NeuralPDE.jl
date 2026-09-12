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

Using physics-informed neural networks.

## Copy-Pasteable Code

```@example poisson
using NeuralPDE, Lux, OptimizationOptimJL, LineSearches, Plots
using DomainSets: Interval

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

# Boundary conditions
bcs = [
    u(0, y) ~ 0.0, u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0
]
# Space domains
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

# Neural network
chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))

# Discretization
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(400; bcs_points = 50))

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

# Optimizer
sol = solve(prob, LBFGS(linesearch = BackTracking()); maxiters = 1000)

xs = ys = 0:0.01:1
analytic_sol_func(x, y) = sinpi(x) * sinpi(y) / (2pi^2)
u_predict = sol(xs, ys; dv = u(x, y))
u_real = [analytic_sol_func(x, y) for x in xs, y in ys]
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real', linetype = :contourf, title = "analytic");
p2 = plot(xs, ys, u_predict', linetype = :contourf, title = "predict");
p3 = plot(xs, ys, diff_u', linetype = :contourf, title = "error");
plot(p1, p2, p3)
```

## Detailed Description

The ModelingToolkit PDE interface for this example looks like this:

```@example poisson
using NeuralPDE, Lux, OptimizationOptimJL, LineSearches, Plots
using DomainSets: Interval

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

# Boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
# Space domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]
```

Here, we define the neural network, where the input of NN equals the number of dimensions and output equals the number of equations in the system.

```@example poisson
# Neural network
chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
```

Here, we build the `PhysicsInformedNN` discretization. The training strategy chooses the
collocation points on which the residuals of the equations are evaluated; quasi-random
sampling with 400 interior points and 50 points per boundary condition works well here.

```@example poisson
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(400; bcs_points = 50))
```

We now define the `PDESystem` and create the PINN problem using the `discretize`
method. `symbolic_discretize` shows the intermediate `ModelingToolkit.System`: the
network parameters `θ_u` are its unknowns, the collocation matrices of each equation are
its parameters, and its costs are the mean squared residuals.

```@example poisson
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
sys = symbolic_discretize(pde_system, discretization)
```

```@example poisson
prob = discretize(pde_system, discretization)
```

The objective is differentiated with Zygote by default; `discretize(pde_system,
discretization; adtype = AutoForwardDiff())` selects another backend, and the section
below compiles the objective and its Enzyme gradient with Reactant. Now we can solve the PDE
using any Optimization.jl optimizer. The result is a `PDENoTimeSolution` that evaluates
the trained network: `sol(x, y; dv = u(x, y))` at points or grids, and `sol[u(x, y)]` on
the default evaluation grid `sol[x]`, `sol[y]`.

```@example poisson
sol = solve(prob, LBFGS(linesearch = BackTracking()); maxiters = 1000)
sol.original_sol.objective
```

We can plot the predicted solution of the PDE and compare it with the analytical solution to plot the relative error.

```@example poisson
xs = ys = 0:0.01:1
analytic_sol_func(x, y) = sinpi(x) * sinpi(y) / (2pi^2)
u_predict = sol(xs, ys; dv = u(x, y))
u_real = [analytic_sol_func(x, y) for x in xs, y in ys]
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real', linetype = :contourf, title = "analytic");
p2 = plot(xs, ys, u_predict', linetype = :contourf, title = "predict");
p3 = plot(xs, ys, diff_u', linetype = :contourf, title = "error");
plot(p1, p2, p3)
```

## Compiling the training loop with Reactant

The generated objective is a batched array program (one network evaluation per finite
difference stencil point over the whole collocation matrix), so it compiles with
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl). The objective and its Enzyme
gradient are compiled once, then used in a plain Optimisers.jl training loop.

```@example poisson
using Reactant, Enzyme, Optimisers

loss = prob.f.f
θ = Reactant.to_rarray(copy(prob.u0))
p = Reactant.to_rarray(prob.p)
compiled_loss = @compile loss(θ, p)
gradient(θ, p) = Enzyme.gradient(Reverse, Const(loss), θ, Const(p))[1]
compiled_gradient = @compile gradient(θ, p)

function train!(θ, p, compiled_gradient; iterations = 500)
    opt_state = Optimisers.setup(Adam(0.01), θ)
    for _ in 1:iterations
        g = compiled_gradient(θ, p)
        opt_state, θ = Optimisers.update(opt_state, θ, g)
    end
    return θ
end
θ = train!(θ, p, compiled_gradient)
compiled_loss(θ, p)
```

The trained parameters can be wrapped back into the solution interface with `remake`
and a zero-iteration solve, or evaluated directly through the network:

```@example poisson
sol_reactant = solve(remake(prob; u0 = Array(θ)), LBFGS(linesearch = BackTracking()); maxiters = 200)
maximum(abs, sol_reactant(xs, ys; dv = u(x, y)) .- u_real)
```
