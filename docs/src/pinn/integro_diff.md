
# Integro Differential Equations

The integral of function u(x),

```math
\int_{0}^{t}u(x)dx
```

where x is variable of integral and t is variable of integro differential equation,

is defined as

```julia
using ModelingToolkit
@parameters t
@variables i(..)
Ii = Symbolics.Integral(t in DomainSets.ClosedInterval(0, t))
```

In multidimensional case,

```julia
Ix = Integral((x,y) in DomainSets.UnitSquare())
```

The UnitSquare domain ranges both x and y from 0 to 1.
Similarly a rectangular or cuboidal domain can be defined using `ProductDomain` of ClosedIntervals.

```julia
Ix = Integral((x,y) in DomainSets.ProductDomain(ClosedInterval(0 ,1), ClosedInterval(0 ,x)))
```


## 1-dimensional example

Lets take an example of an integro differential equation:
```math
\frac{∂}{∂x} u(x)  + 2u(x) + 5 \int_{0}^{x}u(t)dt = 1 for x \geq 0
```
and boundary condition
```math
u(0) = 0
```

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, DomainSets
import ModelingToolkit: Interval, infimum, supremum

@parameters t
@variables i(..)
Di = Differential(t)
Ii = Integral(t in DomainSets.ClosedInterval(0, t))
eq = Di(i(t)) + 2*i(t) + 5*Ii(i(t)) ~ 1
bcs = [i(0.) ~ 0.0]
domains = [t ∈ Interval(0.0,2.0)]
chain = Chain(Dense(1,15,Flux.σ),Dense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy_ = GridTraining(0.05)
discretization = PhysicsInformedNN(chain,
                                   strategy_;
                                   init_params = nothing,
                                   phi = nothing,
                                   derivative = nothing)
pde_system = PDESystem(eq,bcs,domains,[t],[i])
prob = NeuralPDE.discretize(pde_system,discretization)
cb = function (p,l)
    println("Current loss is: $l")
    return false
end
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=100)
```

Plotting the final solution and analytical solution

```julia
ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
phi = discretization.phi
u_predict  = [first(phi([t],res.minimizer)) for t in ts]

analytic_sol_func(t) = 1/2*(exp(-t))*(sin(2*t))
u_real  = [analytic_sol_func(t) for t in ts]
using Plots
plot(ts ,u_real, label = "Analytical Solution")
plot!(ts, u_predict, label = "PINN Solution")
```

![IDE](https://user-images.githubusercontent.com/12683885/129749371-18b44bbc-18c8-49c5-bf30-0cd97ecdd977.png)
