# Convolution integrals with a dummy variable

An integration variable can be local to an `Integral`. It does not need a domain
in the `PDESystem`, an entry in its independent variables, or an extra neural
network input. For example, a convolution acting on `x(t)` can be written as:

```@example integral_bound_variables
using NeuralPDE, ModelingToolkit, DomainSets, Lux

@parameters t tau
@variables x(..)
Dt = Differential(t)
Dtau = Differential(tau)
I = Integral(tau in ClosedInterval(0.0, t))
K(s) = exp(-s) * cos(s)

eq = Dt(Dt(x(t))) + I(K(t - tau) * Dtau(x(tau))) + x(t) ~ 0
bcs = [x(0.0) ~ 0.0, Dt(x(0.0)) ~ 0.0]
domains = [t ∈ Interval(0.0, 60.0)]
@named system = PDESystem(eq, bcs, domains, [t], [x(t)])

chain = Chain(Dense(1, 4), Dense(4, 1))
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(10))
problem = discretize(system, discretization)
nothing # hide
```

Within the integrand, `t` retains the evaluation time while `tau` runs over the
quadrature points. Each occurrence of `x(tau)` evaluates the same network at
`tau`; `x(t)` in the same integrand evaluates it at `t`. Differentiate `x(tau)`
with respect to `tau` to express the derivative of the solution at the quadrature
point. The upper limit `t` is evaluated before quadrature starts.
