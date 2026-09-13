# Optimizing Parameters (Solving Inverse Problems) with Physics-Informed Neural Networks (PINNs)

Consider a Lorenz System,

```math
\begin{align*}
    \frac{\mathrm{d} x}{\mathrm{d}t} &= \sigma (y -x) \, ,\\
    \frac{\mathrm{d} y}{\mathrm{d}t} &= x (\rho - z) - y \, ,\\
    \frac{\mathrm{d} z}{\mathrm{d}t} &= x y - \beta z \, ,\\
\end{align*}
```

with Physics-Informed Neural Networks. Now we would consider the case where we want to optimize the parameters $\sigma$, $\beta$, and $\rho$.

We start by defining the problem,

```@example param_estim
using NeuralPDE, Lux, OptimizationOptimJL, OrdinaryDiffEq, Plots, LineSearches
using DomainSets: Interval

@parameters t, σ_, β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]
```

And the neural networks as,

```@example param_estim
n = 8
chains = [Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, n, σ), Dense(n, 1)) for _ in 1:3]
```

We will add another loss term based on the data that we have to optimize the parameters.

Here we simply calculate the solution of the Lorenz system with [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/#Example-2:-Solving-Systems-of-Equations) based on the adaptivity of the ODE solver. This is used to introduce non-uniformity to the time series.

```@example param_estim
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
ode_prob = ODEProblem(lorenz!, u0, tspan)
ode_sol = solve(ode_prob, Tsit5(), dt = 0.1)
ts = collect(0.0:0.01:1.0)
data = Array(ode_sol(ts))
t_ = reshape(ts, 1, :)
```

Then we define the additional loss function `additional_loss(phi, θ, p)`, the function has
three arguments:

  - `phi`: a `NamedTuple` of batched trial solutions keyed by dependent variable name;
    `phi.x(t_, θ.x)` evaluates the network of `x` on the matrix of points `t_`,
  - `θ`: a `NamedTuple` of the network parameter vectors with the same keys,
  - `p`: the values of the `PDESystem` parameters (`[σ_, ρ, β]` here), which are being
    optimized together with the networks.

```@example param_estim
function additional_loss(phi, θ, p)
    return sum(abs2, phi.x(t_, θ.x) .- data[1:1, :]) / length(ts) +
        sum(abs2, phi.y(t_, θ.y) .- data[2:2, :]) / length(ts) +
        sum(abs2, phi.z(t_, θ.z) .- data[3:3, :]) / length(ts)
end
```

Then finally defining and optimizing using the `PhysicsInformedNN` interface. With
`param_estim = true` the parameters of the `PDESystem` become unknowns of the generated
`System`, initialized from `initial_conditions`.

```@example param_estim
discretization = PhysicsInformedNN(
    chains, GridTraining(0.01); param_estim = true, additional_loss
)
@named pde_system = PDESystem(
    eqs, bcs, domains, [t], [x(t), y(t), z(t)], [σ_, ρ, β];
    initial_conditions = Dict([p => 1.0 for p in [σ_, ρ, β]])
)
prob = discretize(pde_system, discretization)
sol = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 1000)
p_ = [sol.original_sol[σ_], sol.original_sol[ρ], sol.original_sol[β]] # p_ ≈ [10.0, 28.0, 2.667]
```

And then finally some analysis by plotting.

```@example param_estim
ts_plot = 0.0:0.001:1.0
u_predict = [sol(ts_plot; dv = dv) for dv in [x(t), y(t), z(t)]]
plot(ode_sol)
plot!(ts_plot, u_predict, label = ["x(t)" "y(t)" "z(t)"])
```
