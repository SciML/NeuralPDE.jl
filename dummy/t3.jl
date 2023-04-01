using NeuralPDE, Flux, OptimizationOptimisers

linear(u, p, t) = cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)