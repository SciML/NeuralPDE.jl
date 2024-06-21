# Physics Informed Neural Operator for ODEs Solvers

This tutorial provides an example of how to use the Physics Informed Neural Operator (PINO) for solving a family of parametric ordinary differential equations (ODEs).

## Operator Learning  for a family of parametric ODE.

In this section, we will define a parametric ODE and solve it using a PINO. The PINO will be trained to learn the mapping from the parameters of the ODE to its solution.

```@example pino
using Test
using OptimizationOptimisers
using Lux
using Statistics, Random
using NeuralPDE

equation = (u, p, t) -> cos(p * t)
tspan = (0.0f0, 1.0f0)
u0 = 1.0f0
prob = ODEProblem(equation, u0, tspan)

# Define the architecture of the neural network that will be used as the PINO.
branch = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast),
    Lux.Dense(10, 10))
trunk = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast))
deeponet = DeepONet(branch, trunk; linear = nothing)

bounds = [(0.1f0, pi)]
number_of_parameters = 50
db = (bounds.[1][2] - bounds.[1][1]) / 50
dt = (tspan[2] - tspan[1]) / 40
strategy = GridTraining(dt)
strategy = QuasiRandomTraining(points)
opt = OptimizationOptimisers.Adam(0.03)
alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
sol = solve(prob, alg, verbose = false, maxiters = 2000)
predict = sol.u
```

Now let's compare the prediction from the learned operator with the ground truth solution which is obtained by analytic solution the parametric ODE. Where 
Compare prediction with ground truth.

```@example pino
using Plots
# Compute the ground truth solution for each parameter
ground_analytic = (u0, p, t) -> u0 + sin(p * t) / (p)
p_ = bounds[1][1]:strategy.dx:bounds[1][2]
p = reshape(p_, 1, size(p_)[1], 1)
ground_solution = ground_analytic.(u0, p, sol.t.trunk)

# Plot the predicted solution and the ground truth solution as a filled contour plot
# sol.u[1, :, :], represents the predicted solution for each parameter value and time
plot(predict[1, :, :], linetype = :contourf)
plot!(ground_solution[1, :, :], linetype = :contourf)
```

```@example pino
# 'i' is the index of the parameter 'p' in the dataset 
i = 20
# 'predict' is the predicted solution from the PINO model
plot(predict[1, i, :], label = "Predicted")
# 'ground' is the ground truth solution
plot!(ground_solution[1, i, :], label = "Ground truth")
```
