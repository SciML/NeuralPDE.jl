# Imposing Constraints on Physics-Informed Neural Network (PINN) Solutions

Let's consider the Fokker-Planck equation:

```math
- \frac{∂}{∂x} \left [ \left( \alpha x - \beta x^3\right) p(x)\right ] + \frac{\sigma^2}{2} \frac{∂^2}{∂x^2} p(x) = 0 \, ,
```

which must satisfy the normalization condition:

```math
\Delta t \, p(x) = 1
```

with the boundary conditions:

```math
p(-2.2) = p(2.2) = 0
```

with Physics-Informed Neural Networks.

The normalization condition is not a pointwise equation, so it enters the problem as an
`additional_loss`. Its signature is `additional_loss(phi, θ, p)`: `phi` is a `NamedTuple`
of batched network evaluators keyed by dependent variable name, `θ` a `NamedTuple` of the
corresponding parameter vectors and `p` the vector of `PDESystem` parameter values.
`phi.p(X, θ.p)` evaluates the network for `p(x)` on a matrix of points with one column
per point.

```@example fokkerplank
using NeuralPDE, Lux, OptimizationOptimJL, LineSearches, Integrals
using DomainSets: Interval

# the example is taken from this article https://arxiv.org/abs/1910.10503
@parameters x
@variables p(..)
Dx = Differential(x)
Dxx = Differential(x)^2

α = 0.3
β = 0.5
_σ = 0.5
x_0 = -2.2
x_end = 2.2
# Discretization
dx = 0.01

eq = Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))

# Initial and boundary conditions
bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(x_0, x_end)]

# Neural network
inn = 18
chain = Chain(Dense(1, inn, σ), Dense(inn, inn, σ), Dense(inn, inn, σ), Dense(inn, 1))

# Trapezoidal rule for the normalization constraint.
norm_xs = reshape(collect(range(x_0, x_end; length = 200)), 1, :)
norm_dx = norm_xs[2] - norm_xs[1]
function norm_loss_function(phi, θ, p)
    norm_val = sum(phi.p(norm_xs, θ.p)) * norm_dx
    return abs2(norm_val - 1)
end

discretization = PhysicsInformedNN(
    chain, QuadratureTraining(; quadrature_alg = GaussLegendre(n = 200));
    additional_loss = norm_loss_function
)

@named pdesystem = PDESystem(eq, bcs, domains, [x], [p(x)])
sys = symbolic_discretize(pdesystem, discretization)
```

The additional loss is the last cost of the generated `System`. The costs can be
monitored individually from a callback; see the [systems tutorial](@ref systems) for how
to evaluate residuals. Here we simply train:

```@example fokkerplank
prob = discretize(pdesystem, discretization)
sol = solve(prob, BFGS(linesearch = BackTracking()); maxiters = 600)
sol.original_sol.objective
```

And some analysis:

```@example fokkerplank
using Plots
C = 142.88418699042 #fitting param
analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))

xs = x_0:0.01:x_end
u_real = [analytic_sol_func(x) for x in xs]
u_predict = sol(xs; dv = p(x))

plot(xs, u_real, label = "analytic")
plot!(xs, u_predict, label = "predict")
```

## Boundary conditions as constraints of the optimization problem

By default every boundary condition becomes a penalty cost (its mean squared residual),
which is the standard PINN formulation. With `boundary_policy = :constraints` the pointwise
boundary residuals are kept as equality constraints of the `System` instead, so that a
constrained optimizer can enforce them exactly:

```@example fokkerplank
constrained = PhysicsInformedNN(
    chain, QuadratureTraining(; quadrature_alg = GaussLegendre(n = 200));
    additional_loss = norm_loss_function, boundary_policy = :constraints
)
csys = symbolic_discretize(pdesystem, constrained)
length(ModelingToolkit.constraints(csys))
```
