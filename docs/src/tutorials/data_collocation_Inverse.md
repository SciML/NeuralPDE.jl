# Model Improvement in Physics-Informed Neural Networks for solving Inverse problems in ODEs.

Consider an Inverse problem setting for the  [lotka volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations). Here we want to optimize parameters $\alpha$, $\beta$, $\gamma$ and $\delta$ and also solve a parametric Lotka Volterra system.
PINNs are especially useful in these types of problems and are preferred over conventional solvers, due to their ability to learn from observations - the underlying physics governing the distribution of observations.

We start by defining the problem, with a random and non informative initialization for parameters:

```@example improv_param_estim
using NeuralPDE, OrdinaryDiffEq, Lux, Random, OptimizationOptimJL, LineSearches,
      Distributions, Plots
using FastGaussQuadrature
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
initialization = [-5.0, 8.0, 5.0, -7.0]
prob = ODEProblem(lv, u0, tspan, initialization)
```

We require a set of observations before we train the PINN.
Considering we want robust results even for cases where measurement values are sparse and limited in number.
We simulate a system that uses the true parameter `true_p` values and record phenomena/solution (`u`) values algorithmically at only `N=20` pre-decided timepoints in the system's time domain.

The value for `N` can be incremented based on the non linearity (~ `N` degree polynomial) in the measured phenomenon, this tutorial's setting shows that even with minimal but systematically chosen data-points we can extract excellent results.

```@example improv_param_estim
true_p = [1.5, 1.0, 3.0, 1.0]
prob_data = remake(prob, p = true_p)

N = 20
x, w = gausslobatto(N)
a = tspan[1]
b = tspan[2]
```

Now scale the weights and the gauss-lobatto/clenshaw-curtis/gauss-legendre quadrature points to fit in `tspan`.

```@example improv_param_estim
t = map((x) -> (x * (b - a) + (b + a)) / 2, x)
W = map((x) -> x * (b - a) / 2, w)
```

We now have our dataset of `20` measurements in our `tspan` and corresponding weights. Using this we can now use the Data Quadrature loss function by passing `estim_collocate` = `true` in [`NNODE`](@ref).

```@example improv_param_estim
sol_data = solve(prob_data, Tsit5(); saveat = t)
t_ = sol_data.t
u_ = sol_data.u
u1_ = [u_[i][1] for i in eachindex(t_)]
u2_ = [u_[i][2] for i in eachindex(t_)]
dataset = [u1_, u2_, t_, W]
```

Now, let's define a neural network for the PINN using [Lux.jl](https://lux.csail.mit.edu/).

```@example improv_param_estim
rng = Random.default_rng()
Random.seed!(rng, 0)
n = 7
chain = Chain(Dense(1, n, tanh), Dense(n, n, tanh), Dense(n, 2))
ps, st = Lux.setup(rng, chain) |> f64
```

!!! note
    
    While solving Inverse problems, when we specify `param_estim = true` in [`NNODE`](@ref) or [`BNNODE`](@ref), an L2 loss function measuring how the neural network's predictions fit the provided `dataset` is used internally during Maximum Likelihood Estimation.
    Therefore, the `additional_loss` mentioned in the [ODE parameter estimation tutorial](https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode_parameter_estimation/) is not limited to an L2 loss function against data.

We now define the optimizer and [`NNODE`](@ref) - the ODE solving PINN algorithm, for the old PINN model and the proposed new PINN formulation which uses a Data Quadrature loss.
This optimizer and respective algorithms are plugged into the `solve` calls for comparing results between the new and old PINN models.

```@example improv_param_estim
opt = LBFGS(linesearch = BackTracking())

alg_old = NNODE(
    chain, opt; strategy = GridTraining(0.01), dataset = dataset, param_estim = true)

alg_new = NNODE(chain, opt; strategy = GridTraining(0.01), param_estim = true,
    dataset = dataset, estim_collocate = true)
```

Now we have all the pieces to solve the optimization problem.

```@example improv_param_estim
sol_old = solve(
    prob, alg_old; verbose = true, abstol = 1e-12, maxiters = 5000, saveat = 0.01)

sol_new = solve(
    prob, alg_new; verbose = true, abstol = 1e-12, maxiters = 5000, saveat = 0.01)

sol = solve(prob_data, Tsit5(); saveat = 0.01)
sol_points = hcat(sol.u...)
sol_old_points = hcat(sol_old.u...)
sol_new_points = hcat(sol_new.u...)
```

Let's plot the predictions from the PINN models, data used and compare it to the ideal system solution.
First the old model.

```@example improv_param_estim
plot(sol, labels = ["u1" "u2"])
plot!(sol_old, labels = ["u1_pinn_old" "u2_pinn_old"])
scatter!(sol_data, labels = ["u1_data" "u2_data"])
```

Clearly the old model cannot optimize given a realistic, tougher initialization of parameters especially with such limited data. It only seems to work when initial values are close to `true_p` and we have around `500` points for our `tspan`, as seen in the [ODE parameter estimation tutorial](https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode_parameter_estimation/).

Lets move on to the proposed new model...

```@example improv_param_estim
plot(sol, labels = ["u1" "u2"])
plot!(sol_new, labels = ["u1_pinn_new" "u2_pinn_new"])
scatter!(sol_data, labels = ["u1_data" "u2_data"])
```

We can see that it is a good fit! Now let's see what the estimated parameters of the equation tell us in both cases.

```@example improv_param_estim
sol_old.k.u.p
@test any(true_p .- sol_old.k.u.p .> 0.5 .* true_p) # hide
```

Nowhere near the true [1.5, 1.0, 3.0, 1.0]. But the new model gives :

```@example improv_param_estim
sol_new.k.u.p
@test sol_new.k.u.p≈true_p rtol=2e-2 norm=Base.Fix1(maximum, abs) # hide
```

This is indeed close to the true values [1.5, 1.0, 3.0, 1.0].

!!! note
    
    This feature for using a Data collocation loss is also available for BPINNs solving Inverse problems in ODEs. Use a `dataset` of form as described in this tutorial and set `estim_collocate`=`true` and you are good to go.
