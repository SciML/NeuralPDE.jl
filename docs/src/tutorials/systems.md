# [Defining Systems of PDEs for Physics-Informed Neural Networks (PINNs)](@id systems)

In this example, we will solve the PDE system:

```math
\begin{align*}
∂_t^2 u_1(t, x) & = ∂_x^2 u_1(t, x) + u_3(t, x) \, \sin(\pi x) \, ,\\
∂_t^2 u_2(t, x) & = ∂_x^2 u_2(t, x) + u_3(t, x) \, \cos(\pi x) \, ,\\
0 & = u_1(t, x) \sin(\pi x) + u_2(t, x) \cos(\pi x) - e^{-t} \, ,
\end{align*}
```

with the initial conditions:

```math
\begin{align*}
u_1(0, x) & = \sin(\pi x) \, ,\\
∂_t u_1(0, x) & = - \sin(\pi x) \, ,\\
u_2(0, x) & = \cos(\pi x) \, ,\\
∂_t u_2(0, x) & = - \cos(\pi x) \, ,
\end{align*}
```

and the boundary conditions:

```math
\begin{align*}
u_1(t, 0) & = u_1(t, 1) = 0 \, ,\\
u_2(t, 0) & = - u_2(t, 1) = e^{-t} \, ,
\end{align*}
```

with physics-informed neural networks.

## Solution

```@example system
using NeuralPDE, Lux, OptimizationOptimJL, OptimizationOptimisers, LineSearches
using DomainSets: Interval

@parameters t, x
@variables u1(..), u2(..), u3(..)
Dt = Differential(t)
Dtt = Differential(t)^2
Dx = Differential(x)
Dxx = Differential(x)^2

eqs = [
    Dtt(u1(t, x)) ~ Dxx(u1(t, x)) + u3(t, x) * sinpi(x),
    Dtt(u2(t, x)) ~ Dxx(u2(t, x)) + u3(t, x) * cospi(x),
    0.0 ~ u1(t, x) * sinpi(x) + u2(t, x) * cospi(x) - exp(-t)
]

bcs = [
    u1(0, x) ~ sinpi(x),
    u2(0, x) ~ cospi(x),
    Dt(u1(0, x)) ~ -sinpi(x),
    Dt(u2(0, x)) ~ -cospi(x),
    u1(t, 0) ~ 0.0,
    u2(t, 0) ~ exp(-t),
    u1(t, 1) ~ 0.0,
    u2(t, 1) ~ -exp(-t)
]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

# One neural network per dependent variable
n = 15
chain = [Chain(Dense(2, n, σ), Dense(n, n, σ), Dense(n, 1)) for _ in 1:3]

strategy = QuasiRandomTraining(256; bcs_points = 64)
discretization = PhysicsInformedNN(chain, strategy)

@named pdesystem = PDESystem(eqs, bcs, domains, [t, x], [u1(t, x), u2(t, x), u3(t, x)])
prob = discretize(pdesystem, discretization)

res = solve(prob, Adam(0.01); maxiters = 1000)
prob = remake(prob; u0 = res.original_sol.u)
sol = solve(prob, LBFGS(linesearch = BackTracking()); maxiters = 200)
```

A vector of chains gives one network per dependent variable. A single chain with as many
outputs as dependent variables is also accepted: its `i`-th output then represents the
`i`-th dependent variable and all of them share the same parameters.

## Inspecting the individual losses

The `System` returned by `symbolic_discretize` has one cost per equation and boundary
condition, in the order of `eqs` followed by `bcs`, and the [`NeuralPDE.PINNMetadata`](@ref)
records the lowered residual of each. Any of them can be evaluated on the problem with
SymbolicIndexingInterface, which is the way to monitor them from a callback:

```@example system
using SymbolicIndexingInterface: getu, ProblemState
using Statistics: mean

md = pinn_metadata(prob)
residuals = [getu(prob, block.residual) for block in md.blocks]
callback = function (state, loss)
    if state.iter % 100 == 0
        current = ProblemState(; u = state.u, p = state.p)
        println("iteration $(state.iter): loss = $loss")
        println("  pde losses: ", [mean(abs2, r(current)) for r in residuals[1:3]])
        println("  bc losses:  ", [mean(abs2, r(current)) for r in residuals[4:end]])
    end
    return false
end
res = solve(prob, Adam(0.01); maxiters = 200, callback)
```

## Solution Representation

The result of `solve` is a `PDENoTimeSolution`; every dependent variable is evaluated
through it:

```@example system
using Plots

ts = xs = 0:0.01:1
function analytic_sol_func(t, x)
    [exp(-t) * sinpi(x), exp(-t) * cospi(x), (1 + pi^2) * exp(-t)]
end
dvs = [u1(t, x), u2(t, x), u3(t, x)]
u_real = [[analytic_sol_func(t, x)[i] for t in ts, x in xs] for i in 1:3]
u_predict = [sol(ts, xs; dv = dvs[i]) for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i]) for i in 1:3]

ps = []
for i in 1:3
    p1 = plot(ts, xs, u_real[i]', linetype = :contourf, title = "u$i, analytic")
    p2 = plot(ts, xs, u_predict[i]', linetype = :contourf, title = "predict")
    p3 = plot(ts, xs, diff_u[i]', linetype = :contourf, title = "error")
    push!(ps, plot(p1, p2, p3))
end
```

```@example system
ps[1]
```

```@example system
ps[2]
```

```@example system
ps[3]
```

The trained parameters of each network are the array unknowns of the `System`, so they
can be read from the underlying `OptimizationSolution` symbolically:

```@example system
θ_u1 = sol.original_sol[md.networks[1].θ]
length(θ_u1)
```
