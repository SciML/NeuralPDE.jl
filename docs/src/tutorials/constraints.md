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

```@example fokkerplank
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches
using DomainSets: Interval
using IntervalSets: leftendpoint, rightendpoint
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

eq = Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))

# Initial and boundary conditions
bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(x_0, x_end)]

# Neural network
inn = 18
chain = Lux.Chain(Dense(1, inn, Lux.σ),
    Dense(inn, inn, Lux.σ),
    Dense(inn, inn, Lux.σ),
    Dense(inn, 1))

lb = x_0
ub = x_end
# Use a simple trapezoidal rule for the normalization constraint.
# This avoids AD issues with Integrals.jl's C-based quadrature solvers.
norm_xs = collect(range(lb, ub, length = 200))
norm_dx = Float64(norm_xs[2] - norm_xs[1])
function norm_loss_function(phi, θ, p)
    # Evaluate phi at quadrature points (each point as a 1-element vector)
    s = sum(1:length(norm_xs)) do i
        first(phi([norm_xs[i]], θ))
    end
    norm_val = 0.01 * s * norm_dx
    abs(norm_val - 1)
end

discretization = PhysicsInformedNN(chain,
    QuadratureTraining(),
    additional_loss = norm_loss_function)

@named pdesystem = PDESystem(eq, bcs, domains, [x], [p(x)])
prob = discretize(pdesystem, discretization)
phi = discretization.phi

sym_prob = NeuralPDE.symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
approx_derivative_loss_functions = sym_prob.loss_functions.bc_loss_functions

cb_ = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    println("additional_loss: ", norm_loss_function(phi, p.u, nothing))
    return false
end

res = Optimization.solve(
    prob, BFGS(linesearch = BackTracking()), callback = cb_, maxiters = 600)
```

And some analysis:

```@example fokkerplank
using Plots
C = 142.88418699042 #fitting param
analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))

xs = [leftendpoint(d.domain):0.01:rightendpoint(d.domain) for d in domains][1]
u_real = [analytic_sol_func(x) for x in xs]
u_predict = [first(phi(x, res.u)) for x in xs]

plot(xs, u_real, label = "analytic")
plot!(xs, u_predict, label = "predict")
```
