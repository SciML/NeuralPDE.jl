# Differential Equations with Heterogeneous Domains

A differential equation is said to have heterogeneous domains when its dependent variables depend on different independent variables:

```math
u(x) + w(x, v) = \frac{\partial w(x, v)}{\partial w}
```

Here, we write an arbitrary heterogeneous system:

```@example heterogeneous
using NeuralPDE, DiffEqFlux, Flux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x y
@variables p(..) q(..) r(..) s(..)
Dx = Differential(x)
Dy = Differential(y)

# 2D PDE
eq  = p(x) + q(y) + Dx(r(x, y)) + Dy(s(y, x)) ~ 0

# Initial and boundary conditions
bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
       r(x, -1) ~ 0.f0, r(1, y) ~ 0.0f0,
       s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]

# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
           y ∈ Interval(0.0, 1.0)]

numhid = 3
fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
                        [FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
discretization = NeuralPDE.PhysicsInformedNN(fastchains, QuadratureTraining())

@named pde_system = PDESystem(eq, bcs, domains, [x,y], [p(x), q(y), r(x, y), s(y, x)])
prob = SciMLBase.discretize(pde_system, discretization)
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters=100)
```
