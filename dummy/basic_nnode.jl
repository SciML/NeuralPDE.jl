using NeuralPDE, Flux, OptimizationOptimisers

linear(u, p, t) = cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)


chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))


opt = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, opt)


sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose = true, abstol = 1.0f-6,
            maxiters = 200)