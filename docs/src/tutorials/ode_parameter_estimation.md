# Parameter Estimation with Physics-Informed Neural Networks for ODEs

Consider the [lotka volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)

with Physics-Informed Neural Networks. Now we would consider the case where we want to optimize the parameters $\alpha$, $\beta$, $\gamma$ and $\delta$.

We start by defining the problem:

```@example param_estim_lv
using NeuralPDE, OrdinaryDiffEq
using Lux, Random
using OptimizationOptimJL, LineSearches
using Plots
using Test # hide

function lv(u, p, t)
    u₁, u₂ = u
    α, β, γ, δ = p
    du₁ = α * u₁ - β * u₁ * u₂
    du₂ = δ * u₁ * u₂ - γ * u₂
    [du₁, du₂]
end

tspan = (0.0, 5.0)
u0 = [5.0, 5.0]
prob = ODEProblem(lv, u0, tspan, [1.0, 1.0, 1.0, 1.0])
```

As we want to estimate the parameters as well, let's get some data.

```@example param_estim_lv
true_p = [1.5, 1.0, 3.0, 1.0]
prob_data = remake(prob, p = true_p)
sol_data = solve(prob_data, Tsit5(), saveat = 0.01)
t_ = sol_data.t
u_ = reduce(hcat, sol_data.u)
```

Now, let's define a neural network for the PINN using [Lux.jl](https://lux.csail.mit.edu/).

```@example param_estim_lv
rng = Random.default_rng()
Random.seed!(rng, 0)
n = 15
chain = Lux.Chain(
    Lux.Dense(1, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, 2)
)
ps, st = Lux.setup(rng, chain) |> Lux.f64
```

Next we define an additional loss term to in the total loss which measures how the neural network's predictions is fitting the data.

```@example param_estim_lv
function additional_loss(phi, θ)
    return sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)
end
```

Next we define the optimizer and [`NNODE`](@ref) which is then plugged into the `solve` call.

```@example param_estim_lv
opt = LBFGS(linesearch = BackTracking())
alg = NNODE(chain, opt, ps; strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 500),
    param_estim = true, additional_loss = additional_loss)
```

Now we have all the pieces to solve the optimization problem.

```@example param_estim_lv
sol = solve(prob, alg, verbose = true, abstol = 1e-8, maxiters = 5000, saveat = t_)
@test sol.k.u.p≈true_p rtol=1e-2 # hide
```

Let's plot the predictions from the PINN and compare it to the data.

```@example param_estim_lv
plot(sol, labels = ["u1_pinn" "u2_pinn"])
plot!(sol_data, labels = ["u1_data" "u2_data"])
```

We can see it is a good fit! Now let's see if we have the parameters of the equation also estimated correctly or not.

```@example param_estim_lv
sol.k.u.p
```

We can see it is indeed close to the true values [1.5, 1.0, 3.0, 1.0].
