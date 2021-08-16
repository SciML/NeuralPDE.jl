# Integro Differential Equations
Lets take an example of an integro differential equation :
```math
\frac{∂}{∂x} u(x)  + 2u(x) + 5 \int_{0}{x}u(t)dt = 1 for x \geq 0
```
and boundary condition
```math
u(0) = 0
```

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, DomainSets
import ModelingToolkit: Interval, infimum, supremum
```
```julia
@parameters t
@variables i(..)
Di = Differential(t)
```
We define the `Differential` and parameters usually, and then we define the integral  with first parameter as the integrating variable, if multiple integrating variables are present in the domain, pass the vector in order of integration. And the second parameter is the `VarDomainPairing` that defines the domain of the integration.

```julia
Ii = Integral(t, t in DomainSets.ClosedInterval(0, t))
```
Then we define the equation and boundary conditions. Note that the integrand is passed inside the defined `Integral` operator.
```julia
eq = Di(i(t)) + 2*i(t) + 5*Ii(i(t)) ~ 1
bcs = [i(0.) ~ 0.0]
```
We define the domain and the neural network solution `u`
```julia
domains = [t ∈ Interval(0.0,2.0)]
chain = Chain(Dense(1,15,Flux.σ),Dense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
```
We use the grid training strategy and then use the `PhysicsInformedNN` interface to define the optimization problem.
```julia
strategy_ = NeuralPDE.GridTraining(0.05)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy_;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             )
pde_system = PDESystem(eq,bcs,domains,[t],[i])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=100)
```
Taking the time range from the domain,
```julia
ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]
```
And then take the optimised solution
```julia
phi = discretization.phi
u_predict  = [first(phi([t],res.minimizer)) for t in ts]
```
The analytical solution to the above equation is
```math
\frac{1}{2}e^{-t}sin(2t)
```
```julia
analytic_sol_func(t) = 1/2*(exp(-t))*(sin(2*t))
u_real  = [analytic_sol_func(t) for t in ts]
```
Plotting the final solution and analytical solution
```julia
using Plots
plot(u_real, ts, label = "Analytical Solution")
plot!(u_predict, ts, label = "PINN Solution")
```
