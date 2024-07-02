# Physics Informed Neural Operator for ODEs Solvers

This tutorial provides an example of how to use the Physics Informed Neural Operator (PINO) for solving a family of parametric ordinary differential equations (ODEs).

## Operator Learning for a family of parametric ODEs

In this section, we will define a parametric ODE and solve it using a PINO. The PINO will be trained to learn the mapping from the parameters of the ODE to its solution.

```@example pino
using Test
using OptimizationOptimisers
using Lux
using Statistics, Random
using LuxNeuralOperators
using NeuralPDE

equation = (u, p, t) -> p[1] * cos(p[2] * t) + p[3]
tspan = (0.0f0, 1.0f0)
u0 = 1.0f0
prob = ODEProblem(equation, u0, tspan)

number_of_parameter = 3
deeponet = LuxNeuralOperators.DeepONet(
    Chain(
        Dense(number_of_parameter => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast), Dense(10 => 10)),
    Chain(Dense(1 => 10, Lux.tanh_fast), Dense(10 => 10, Lux.tanh_fast),
        Dense(10 => 10, Lux.tanh_fast)))

u = rand(3, 50)
v = rand(1, 40, 1)
θ, st = Lux.setup(Random.default_rng(), deeponet)
c = deeponet((u, v), θ, st)[1]

bounds = [(1.0f0, pi), (1.0f0, 2.0f0), (2.0f0, 3.0f0)]
number_of_parameters = 50
strategy = StochasticTraining(40)
opt = OptimizationOptimisers.Adam(0.03)
alg = PINOODE(deeponet, opt, bounds, number_of_parameters; strategy = strategy)
sol = solve(prob, alg, verbose = true, maxiters = 3000)
```

Now let's compare the prediction from the learned operator with the ground truth solution which is obtained by analytic solution the parametric ODE. Where 
Compare prediction with ground truth.

```@example pino
using Plots

function get_trainset(bounds, tspan , number_of_parameters, dt)
    p_ = [range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]
    p = vcat([collect(reshape(p_i, 1, size(p_i,1))) for p_i in p_]...)
    t_ = collect(tspan[1]:dt:tspan[2])
    t = collect(reshape(t_, 1, size(t_, 1), 1))
    (p,t)
end

# Compute the ground truth solution for each parameter
ground_solution = (u0, p, t) -> u0 + p[1] / p[2] * sin(p[2] * t) + p[3]*t
function ground_solution_f(p,t)
    reduce(hcat,[[ground_solution(u0, p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
end

(p,t) = get_trainset(bounds, tspan, 50, 0.025f0)
ground_solution_ = ground_solution_f(p,t)
predict = sol.interp((p,t))

# Calculate the mean error and  the standard deviation of the errors
errors = ground_solution_ - predict
mean_error = mean(errors)
std_error = std(errors)

# generate the solution with new parameters for test the model
p,t = get_trainset(bounds, tspan, 100,  0.01f0) 
ground_solution_ = ground_solution_f(p,t)
predict = sol.interp((p,t))

errors = ground_solution_ - predict
mean_error = mean(errors)
std_error = std(errors)

# Plot the predicted solution and the ground truth solution as a filled contour plot
# predict, represents the predicted solution for each parameter value and time
plot(predict, linetype = :contourf)
plot!(ground_solution_, linetype = :contourf)
```

```@example pino
# 'i' is the index of the parameter 'p' in the dataset 
i = 5
# 'predict' is the predicted solution from the PINO model
plot(predict[:, i], label = "Predicted")
# 'ground' is the ground truth solution
plot!(ground_solution_[:, i], label = "Ground truth")
```


