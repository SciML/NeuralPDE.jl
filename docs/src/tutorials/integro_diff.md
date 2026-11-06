# Solving Integro-Differential Equations

`Symbolics.Integral` terms can appear in the equations of a `PDESystem`. The integral

```math
\int_{0}^{t}u(\tau)d\tau
```

over the integrating variable ``\tau``, whose upper limit is the independent variable
``t``, is written as

```julia
using ModelingToolkit, DomainSets
@parameters t tau
@variables u(..)
I = Integral(tau in DomainSets.ClosedInterval(0, t))
I(u(tau))
```

Limits may depend on the independent variables, integrals may be nested, and several
integrating variables can be combined with a `ProductDomain`:

```julia
@parameters x y
Ixy = Integral(
    (x, y) in DomainSets.ProductDomain(
        DomainSets.ClosedInterval(0, 1), DomainSets.ClosedInterval(0, x)
    )
)
```

`discretize` replaces each integral with a quadrature over the network, evaluated at
every collocation point. The quadrature rule is the `integral_alg` keyword of
[`PhysicsInformedNN`](@ref).

## 1-dimensional example

Consider the integro-differential equation

```math
\frac{d}{dt} i(t) + 2i(t) + 5 \int_{0}^{t}i(\tau)d\tau = 1 \ \text{for} \ t \geq 0
```

with the initial condition

```math
i(0) = 0
```

and the analytical solution ``i(t) = \frac{1}{2} e^{-t} \sin(2t)``.

```@example integro
using NeuralPDE, Lux, ModelingToolkit, OptimizationOptimJL, LineSearches, Plots
using DomainSets: ClosedInterval, Interval
using OptimizationOptimisers: Adam
using Random: Xoshiro

@parameters t tau
@variables i(..)
Di = Differential(t)
Ii = Integral(tau in ClosedInterval(0.0, t))
eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(tau)) ~ 1
bcs = [i(0.0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]
chain = Chain(Dense(1, 15, σ), Dense(15, 1))

discretization = PhysicsInformedNN(chain, GridTraining(0.05); rng = Xoshiro(110))
@named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])
prob = discretize(pde_system, discretization)
```

Equations with `Integral` terms are differentiated with `AutoZygote()` by default
(see [`default_adtype`](@ref NeuralPDE.default_adtype)), because Enzyme cannot yet
differentiate the quadrature objective. We train with Adam and then refine with BFGS:

```@example integro
res = solve(prob, Adam(0.01); maxiters = 200)
prob = remake(prob; u0 = res.original_sol.u)
sol = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 300)
```

Plotting the trained network against the analytical solution:

```@example integro
ts = sol[t]
i_predict = sol[i(t)]

analytic_sol_func(t) = 1 / 2 * exp(-t) * sin(2 * t)
i_real = analytic_sol_func.(ts)
plot(ts, i_real, label = "Analytical Solution")
plot!(ts, i_predict, label = "PINN Solution")
```

```@example integro
using Test # hide
@test maximum(abs, i_predict .- i_real) < 0.05 # hide
nothing # hide
```
