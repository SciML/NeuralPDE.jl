# Bayesian Physics informed Neural Network ODEs Solvers

Bayesian inference for PINNs provides an approach to ODE solution finding and parameter estimation with quantified uncertainty.

## The Lotka-Volterra Model

The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations. These differential equations are frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey. The populations change through time according to the pair of equations:

```math
\begin{aligned}
\frac{\mathrm{d}x}{\mathrm{d}t} &= (\alpha - \beta y(t))x(t), \\
\frac{\mathrm{d}y}{\mathrm{d}t} &= (\delta x(t) - \gamma)y(t)
\end{aligned}
```

where $x(t)$ and $y(t)$ denote the populations of prey and predator at time $t$, respectively, and $\alpha, \beta, \gamma, \delta$ are positive parameters.

We implement the Lotka-Volterra model and simulate it with ideal parameters $\alpha = 1.5$, $\beta = 1$, $\gamma = 3$, and $\delta = 1$ and initial conditions $x(0) = y(0) = 1$.

We then solve the equations and estimate the parameters of the model with priors for $\alpha$, $\beta$, $\gamma$ and $\delta$ as  `Normal(1,2)`, `Normal(2,2)`, `Normal(2,2)` and `Normal(0,2)` using a neural network.

```@example bpinn
using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Distributions, Random

function lotka_volterra(u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    dx = (α - β * y) * x # prey
    dy = (δ * x - γ) * y # predator

    return [dx, dy]
end

# initial-value problem.
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 4.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)
```

With the [`saveat`](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/) argument, we can specify that the solution is stored only at `saveat` time units.

```@example bpinn
# Solve using OrdinaryDiffEq.jl solver
dt = 0.01
solution = solve(prob, Tsit5(); saveat = dt)
```

We generate noisy observations to use for the parameter estimation task in this tutorial. To make the example more realistic we add random normally distributed noise to the simulation.

```@example bpinn
# Dataset creation for parameter estimation (30% noise)
time = solution.t
u = hcat(solution.u...)
x = u[1, :] + (u[1, :]) .* (0.3 .* randn(length(u[1, :])))
y = u[2, :] + (u[2, :]) .* (0.3 .* randn(length(u[2, :])))
dataset = [x, y, time]

# Plotting the data which will be used
plot(time, x, label = "noisy x")
plot!(time, y, label = "noisy y")
plot!(solution, labels = ["x" "y"])
```

Let's define a PINN.

```@example bpinn
# Neural Networks must have 2 outputs as u -> [dx,dy] in function lotka_volterra()
chain = @closure Chain(Dense(1, 6, tanh), Dense(6, 6, tanh), Dense(6, 2))
```

The dataset we generated can be passed for doing parameter estimation using provided priors in `param` keyword argument for [`BNNODE`](@ref).

```@example bpinn
alg = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [
        Normal(1, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(0, 2)], progress = false)

sol_pestim = solve(prob, alg; saveat = dt)

nothing #hide
```

The solution for the ODE is returned as a nested vector `sol_flux_pestim.ensemblesol`. Here, [$x$ , $y$] would be returned.

```@example bpinn
# plotting solution for x,y for chain
plot(time, sol_pestim.ensemblesol[1], label = "estimated x")
plot!(time, sol_pestim.ensemblesol[2], label = "estimated y")

# comparing it with the original solution
plot!(solution, labels = ["true x" "true y"])
```

We can see the estimated ODE parameters by -

```@example bpinn
sol_pestim.estimated_de_params
```

We can see it is close to the true values of the parameters.
