# Solving DAEs with Physics-Informed Neural Networks (PINNs)

!!! note
    
    It is highly recommended you first read the [solving ordinary differential
    equations with DifferentialEquations.jl tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/) before reading this tutorial.


This tutorial is an introduction to using physics-informed neural networks (PINNs) for solving differential algebraic equations (DAEs). 




## Solving an DAE with PINNs

Let's solve a simple DAE system:

```@example dae
example1 = (du, u, p, t) -> [cos(2pi * t), u[2] + cos(2pi * t)]

u₀ = [1.0, -1.0]
du₀ = [0.0, 0.0]
tspan = (0.0f0, 1.0f0)

differential_vars = [true, false]
prob = DAEProblem(example, du₀, u₀, tspan; differential_vars = differential_vars)
chain = Flux.Chain(Dense(1, 15, cos), Dense(15, 15, sin), Dense(15, 2))
opt = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNDAE(chain, opt; autodiff = false)

sol = solve(prob,
    alg, verbose = false, dt = 1 / 100.0f0,
    maxiters = 3000, abstol = 1.0f-10)
@test ground_sol(0:(1 / 100):1) ≈ sol atol=0.4
```


```@example dae
function example1(du, u, p, t)
    du[1] =  cos(2pi * t)
    du[2] = u[2] + cos(2pi * t)
    nothing
end
M = [1.0 0
    0 0]
f = ODEFunction(example1, mass_matrix = M)
prob_mm = ODEProblem(f, u₀, tspan)
ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)
```


```@example dae
plot(ground_sol, tspan = tspan, layout = (2, 1))
plot!(sol, tspan = tspan, layout = (2, 1))
```