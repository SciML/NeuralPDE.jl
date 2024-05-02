# Physics Informed Neural Operator for ODEs Solvers

This tutorial provides an example of how using Physics Informed Neural Operator (PINO) for solving family of parametric ordinary differential equations (ODEs).

## Operator Learning  for a family of parametric ODE.

```@example pino
using Test
using OrdinaryDiffEq, OptimizationOptimisers
using Lux
using Statistics, Random
using NeuralPDE

equation = (u, p, t) -> cos(p * t)
tspan = (0.0f0, 1.0f0)
u0 = 1.0f0
prob = ODEProblem(equation,  u0, tspan)

# initilize DeepONet operator
branch = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast),
    Lux.Dense(10, 10))
trunk = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast))

deeponet = NeuralPDE.DeepONet(branch, trunk; linear = nothing)

bounds = (p = [0.1f0, pi],)
#TODO add truct
strategy  = (branch_size = 50, trunk_size = 40)
# strategy = (branch_size = 50, dt = 0.1)?
opt = OptimizationOptimisers.Adam(0.03)
alg = NeuralPDE.PINOODE(deeponet, opt, bounds; strategy = strategy)

sol = solve(prob, alg, verbose = true, maxiters = 2000)
```

Now let's compare the prediction from the learned operator with the ground truth solution which is obtained by analytic solution the parametric ODE. Where 
Compare prediction with ground truth.

```@example pino
using Plots
# Compute the ground truth solution for each parameter value and time in the solution
# The '.' operator is used to apply the functd ion element-wise
ground_analytic = (u0, p, t) -> begin u0 + sin(p * t) / (p)
p_ = range(bounds.p[1], stop = bounds.p[2], length = strategy.branch_size)
p = reshape(p_, 1, branch_size, 1)
ground_solution = ground_analytic.(u0, p, sol.t.trunk)

# Plot the predicted solution and the ground truth solution as a filled contour plot
# sol.u[1, :, :], represents the predicted solution for each parameter value and time
plot(sol.u[1, :, :], linetype = :contourf)
plot!(ground_solution[1, :, :], linetype = :contourf)
```


```@example pino
using Plots

# 'i' is the index of the parameter 'a' in the dataset
i = 45

# 'predict' is the predicted solution from the PINO model
# 'ground' is the ground truth solution
plot(predict[1, :, i], label = "Predicted")
plot!(ground[1, :, i], label = "Ground truth")
```
