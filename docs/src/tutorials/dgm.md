## Solving PDEs using Deep Galerkin Method

### Overview

Deep Galerkin Method is a meshless deep learning algorithm to solve high dimensional PDEs. The algorithm does so by approximating the solution of a PDE with a neural network. The loss function of the network is defined in the similar spirit as PINNs, composed of PDE loss and boundary condition loss.

In the following example, we demonstrate computing the loss function using Quasi-Random Sampling, a sampling technique that uses quasi-Monte Carlo sampling to generate low discrepancy random sequences in high dimensional spaces.

### Algorithm

The authors of DGM suggest a network composed of LSTM-type layers that works well for most of the parabolic and quasi-parabolic PDEs.

```math
\begin{align*}
S^1 &= \sigma_1(W^1 \vec{x} + b^1); \\
Z^l &= \sigma_1(U^{z,l} \vec{x} + W^{z,l} S^l + b^{z,l}); \quad l = 1, \ldots, L; \\
G^l &= \sigma_1(U^{g,l} \vec{x} + W^{g,l} S_l + b^{g,l}); \quad l = 1, \ldots, L; \\
R^l &= \sigma_1(U^{r,l} \vec{x} + W^{r,l} S^l + b^{r,l}); \quad l = 1, \ldots, L; \\
H^l &= \sigma_2(U^{h,l} \vec{x} + W^{h,l}(S^l \cdot R^l) + b^{h,l}); \quad l = 1, \ldots, L; \\
S^{l+1} &= (1 - G^l) \cdot H^l + Z^l \cdot S^{l}; \quad l = 1, \ldots, L; \\
f(t, x; \theta) &= \sigma_\text{out}(W S^{L+1} + b).
\end{align*}
```

where $\vec{x}$ is the concatenated vector of $(t, x)$ and $L$ is the number of LSTM type layers in the network.

### Example

Let's try to solve the following Burger's equation using Deep Galerkin Method for $\alpha = 0.05$:

```math
\partial_t u(t, x) + u(t, x) \partial_x u(t, x) - \alpha \partial_{xx} u(t, x) = 0 
```

defined over

```math
t \in [0, 1], x \in [-1, 1] 
```

with boundary conditions

```math
\begin{align*}
u(t, x) & = - \sin(πx), \\
u(t, -1) & = 0, \\
u(t, 1) & = 0
\end{align*}
```

### Copy- Pasteable code

```@example dgm
using NeuralPDE
using ModelingToolkit, Optimization, OptimizationOptimisers
using Distributions
using DomainSets: Interval
using Plots

@parameters x t
@variables u(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Dx^2
α = 0.05
# Burger's equation
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - α * Dxx(u(t, x)) ~ 0

# boundary conditions
bcs = [
    u(0.0, x) ~ -sin(π * x),
    u(t, -1.0) ~ 0.0,
    u(t, 1.0) ~ 0.0
]

domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(-1.0, 1.0)]

# NeuralPDE, using Deep Galerkin Method
strategy = QuasiRandomTraining(256, minibatch = 32)
discretization = DeepGalerkin(2, 1, 50, 5, tanh, tanh, identity, strategy)
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    (p.iter % 20 == 0) && println("$(p.iter) => $l")
    return false
end

res = solve(prob, Adam(0.1); maxiters = 100)
prob = remake(prob, u0 = res.u)
res = solve(prob, Adam(0.01); maxiters = 500)
phi = discretization.phi

tgrid = 0.0:0.01:1.0
xgrid = -1.0:0.01:1.0

u_predict = [first(phi([t, x], res.minimizer)) for t in tgrid, x in xgrid]

# Plot the solution
p1 = plot(tgrid, xgrid, u_predict', linetype = :contourf, title = "DGM Solution");

# Also check initial condition
u_initial = [first(phi([0.0, x], res.minimizer)) for x in xgrid]
u_expected = [-sin(π * x) for x in xgrid]
p2 = plot(xgrid, u_initial, label = "DGM at t=0", title = "Initial Condition");
plot!(p2, xgrid, u_expected, label = "Expected: -sin(πx)", linestyle = :dash);
plot(p1, p2, layout = (1, 2))
```
