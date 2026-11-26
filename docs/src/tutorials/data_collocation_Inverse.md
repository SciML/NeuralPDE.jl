# Model Improvement in Physics-Informed Neural Networks for solving Inverse problems in ODEs.

Consider an Inverse problem setting for the [lotka volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations). Here we want to optimize parameters $\alpha$, $\beta$, $\gamma$ and $\delta$ and also solve a parametric Lotka Volterra system.
PINNs are especially useful in these types of problems and are preferred over conventional solvers, due to their ability to learn from observations - the underlying physics governing the distribution of observations.

We start by defining the problem, with a random and non informative initialization for parameters:

```@example improv_param_estim
using ModelingToolkit, NeuralPDE, SciMLBase, OrdinaryDiffEq, Lux, Random, OptimizationOptimJL, LineSearches,
      Distributions, Plots
using Optim: LBFGS
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
n = 7
chain = Chain(Dense(1, n, tanh), Dense(n, n, tanh), Dense(n, 2))
ps, _ = Lux.setup(Xoshiro(1), chain) |> f64
```

!!! note
    
    While solving Inverse problems, when we specify `param_estim = true` in [`NNODE`](@ref) or [`BNNODE`](@ref), an L2 loss function measuring how the neural network's predictions fit the provided `dataset` is used internally during Maximum Likelihood Estimation.
    Therefore, the `additional_loss` mentioned in the [ODE parameter estimation tutorial](https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode_parameter_estimation/) is not limited to an L2 loss function against data.

Both fits use the same optimizer, starting ODE parameters, and seeded network weights. Each [`NNODE`](@ref) gets its own copy of the weights.

To illustrate parameter recovery with a sparse physics grid, we use a grid step longer than the time span. The grid evaluates the ODE residual at the initial state, where the four rates appear only through $\alpha - 5\beta$ and $5\delta - \gamma$. Both fits use all 20 observations. The data-collocation fit also evaluates the ODE at the measured states, whose different populations help distinguish the rates.

```@example improv_param_estim
opt = LBFGS(linesearch = BackTracking())
strategy = GridTraining(6.0)

alg_old = NNODE(
    chain, opt, deepcopy(ps); strategy, dataset, param_estim = true
)

alg_new = NNODE(
    chain, opt, deepcopy(ps); strategy, dataset, param_estim = true,
    estim_collocate = true
)
```

We give both fits a budget of 5,000 optimizer iterations. A fit can finish successfully before the limit or use the full budget.

```@example improv_param_estim
maxiters = 5000
sol_old = solve(prob, alg_old; verbose = false, abstol = 1.0e-12, maxiters, saveat = 0.01)
sol_new = solve(prob, alg_new; verbose = false, abstol = 1.0e-12, maxiters, saveat = 0.01)

for fit in (sol_old, sol_new) # hide
    result = fit.k # hide
    @test result.retcode == ReturnCode.Success || # hide
        (result.retcode == ReturnCode.MaxIters && result.stats.iterations == maxiters) # hide
end # hide

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

Now plot the data-collocation fit against the same observations and reference solution.

```@example improv_param_estim
plot(sol, labels = ["u1" "u2"])
plot!(sol_new, labels = ["u1_pinn_new" "u2_pinn_new"])
scatter!(sol_data, labels = ["u1_data" "u2_data"])
```

At this budget and initialization, the baseline misses at least one parameter by more than `50%`, while the data-collocation fit recovers each parameter within `2%`. We measure the absolute relative error for each parameter against `true_p` and require all estimates to be finite.

```@example improv_param_estim
sol_old.k.u.p
```

```@example improv_param_estim
max_relative_parameter_error(estimate) = maximum(abs.((estimate .- true_p) ./ true_p)) # hide
baseline_has_gap(estimate) = all(isfinite, estimate) && max_relative_parameter_error(estimate) > 0.5 # hide
collocation_recovers(estimate) = all(isfinite, estimate) && max_relative_parameter_error(estimate) <= 0.02 # hide
@test baseline_has_gap(sol_old.k.u.p) # hide
```

The data-collocation parameter estimates are shown below.

```@example improv_param_estim
sol_new.k.u.p
```

```@example improv_param_estim
@test collocation_recovers(sol_new.k.u.p) # hide
with_second_parameter(value) = setindex!(copy(true_p), value, 2) # hide
@test !baseline_has_gap(with_second_parameter(Inf)) # hide
@test !baseline_has_gap(with_second_parameter(0.89)) # hide
@test baseline_has_gap(with_second_parameter(1.6)) # hide
@test !collocation_recovers(with_second_parameter(1.05)) # hide
```

!!! note
    
    This feature for using a Data collocation loss is also available for BPINNs solving Inverse problems in ODEs. Use a `dataset` of the form as described in this tutorial and set `estim_collocate`=`true` and you are good to go.
