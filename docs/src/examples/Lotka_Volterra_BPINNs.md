# Bayesian Physics informed Neural Network ODEs Solvers

Bayesian inference for PINNs provides an approach to ODE solution finding and parameter estimation with quantified uncertainty.

## The Lotka-Volterra Model

The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations.
These differential equations are frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey.
The populations change through time according to the pair of equations

```math
\begin{aligned}
\frac{\mathrm{d}x}{\mathrm{d}t} &= (\alpha - \beta y(t))x(t), \\
\frac{\mathrm{d}y}{\mathrm{d}t} &= (\delta x(t) - \gamma)y(t)
\end{aligned}
``` 

where $x(t)$ and $y(t)$ denote the populations of prey and predator at time $t$, respectively, and $\alpha, \beta, \gamma, \delta$ are positive parameters.

We implement the Lotka-Volterra model and simulate it with ideal parameters $\alpha = 1.5$, $\beta = 1$, $\gamma = 3$, and $\delta = 1$ and initial conditions $x(0) = y(0) = 1$. 

We then solve the equations and estimate the parameters of the model with priors for $\alpha$, $\beta$, $\gamma$ and $\delta$ as  Normal(1,2), Normal(2,2), Normal(2,2) and Normal(0,2) using a Flux.jl Neural Network, chain_flux.

And also solve the equations for the constructed ODEProblem's provided ideal `p` values using a Lux.jl Neural Network, chain_lux.

```julia
using NeuralPDE, Flux, Lux, Plots, StatsPlots, OrdinaryDiffEq, Distributions 
 
function lotka_volterra(u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    dx = (α - β * y) * x # prey
    dy = (δ * x - γ) * y # predator

    return [dx, dy]
end

# initial-value problem.
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 6.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

```
With the [`saveat` argument](https://docs.sciml.ai/latest/basics/common_solver_opts/) we can specify that the solution is stored only at `saveat` time units(default saveat=1 / 50.0).

```julia
# Plot solution got by Standard DifferentialEquations.jl ODE solver
solution = solve(prob, Tsit5(); saveat = 0.05)
plot(solve(prob, Tsit5()))

```

We generate noisy observations to use for the parameter estimation tasks in this tutorial.
To make the example more realistic we add random normally distributed noise to the simulation.


```julia
# Dataset creation for parameter estimation (30% noise)
time = solution.t
u = hcat(solution.u...)
x = u[1, :] + (0.3 .*u[1, :]).*randn(length(u[1, :]))
y = u[2, :] + (0.3 .*u[2, :]).*randn(length(u[2, :]))
dataset = [x, y, time]

# Neural Networks must have 2 outputs as u -> [dx,dy] in function lotka_volterra()
chainflux = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh), 
                       Flux.Dense(6, 2)) |> Flux.f64
chainlux = Lux.Chain(Lux.Dense(1, 7, Lux.tanh), Lux.Dense(7, 7, Lux.tanh),
                    Lux.Dense(7, 2))

```
A Dataset is required as parameter estimation is being done using provided priors in `param` keyword argument for BNNODE.

```julia
alg1 = NeuralPDE.BNNODE(chainflux,
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [
        Normal(1, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(0, 2),
    ], progress = true)

sol_flux_pestim = solve(prob, alg1)

# Dataset not needed as we are solving the equation with ideal parameters
alg2 = NeuralPDE.BNNODE(chainlux,
    draw_samples = 1000,
    phystd = [0.05, 0.05],
    priorsNNw = (0.0, 10.0),
    progress = true)

sol_lux = solve(prob, alg2)

#testing timepoints must match keyword arg `saveat`` timepoints of solve() call
t = collect(Float64, prob.tspan[1]:(1 / 50.0):prob.tspan[2]) 

```

the solution for the ODE is retured as a nested vector sol_flux_pestim.ensemblesol.
here, [$x$ , $y$] would be returned
All estimated ode parameters are returned as a vector sol_flux_pestim.estimated_ode_params.
here, [$\alpha$, $\beta$, $\gamma$, $\delta$] 

```julia
# plotting solution for x,y for chain_flux
plot(t,sol_flux_pestim.ensemblesol[1])
plot!(t,sol_flux_pestim.ensemblesol[2])

# estimated ODE parameters by .estimated_ode_params, weights and biases by .estimated_nn_params
println(sol_flux_pestim.estimated_ode_params)
sol_flux_pestim.estimated_nn_params
 
# plotting solution for x,y for chain_lux
plot(t,sol_lux.ensemblesol[1])
plot!(t,sol_lux.ensemblesol[2])

# estimated weights and biases by .estimated_nn_params for chain_lux 
sol_lux.estimated_nn_params

```