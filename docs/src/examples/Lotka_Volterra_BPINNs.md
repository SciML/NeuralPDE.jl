# Bayesian Physics informed Neural Network ODEs Solvers

Most of the scientific community deals with the basic problem of trying to mathematically model the reality around them and this often involves dynamical systems. The general trend to model these complex dynamical systems is through the use of differential equations.
Differential equation models often have non-measurable parameters.
The popular “forward-problem” of simulation consists of solving the differential equations for a given set of parameters, the “inverse problem” to simulation, known as parameter estimation, is the process of utilizing data to determine these model parameters.
Bayesian inference provides a robust approach to parameter estimation with quantified uncertainty.

## The Lotka-Volterra Model

The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations.
These differential equations are frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey.
The populations change through time according to the pair of equations

$$
\begin{aligned}
\frac{\mathrm{d}x}{\mathrm{d}t} &= (\alpha - \beta y(t))x(t), \\
\frac{\mathrm{d}y}{\mathrm{d}t} &= (\delta x(t) - \gamma)y(t)
\end{aligned}
$$

where $x(t)$ and $y(t)$ denote the populations of prey and predator at time $t$, respectively, and $\alpha, \beta, \gamma, \delta$ are positive parameters.

We implement the Lotka-Volterra model and simulate it with parameters $\alpha = 1.5$, $\beta = 1$, $\gamma = 3$, and $\delta = 1$ and initial conditions $x(0) = y(0) = 1$.

```julia
# Define Lotka-Volterra model.
function lotka_volterra(du, u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    du[1] = (α - β * y) * x # prey
    du[2] = (δ * x - γ) * y # predator

    return nothing
end

# Define initial-value problem.
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Plot simulation.
plot(solve(prob, Tsit5()))

solution = solve(prob, Tsit5(); saveat = 0.05)

time = solution.t
u = hcat(solution.u...)
# BPINN AND TRAINING DATASET CREATION, NN create, Reconstruct
x = u[1, :] + 0.5 * randn(length(u[1, :]))
y = u[2, :] + 0.5 * randn(length(u[1, :]))
dataset = [x[1:50], y[1:50], time[1:50]]

# NN has 2 outputs as u -> [dx,dy]
chainlux1 = Lux.Chain(Lux.Dense(1, 6, Lux.tanh), Lux.Dense(6, 6, Lux.tanh),
    Lux.Dense(6, 2))
chainflux1 = Flux.Chain(Flux.Dense(1, 6, tanh), Flux.Dense(6, 6, tanh), Flux.Dense(6, 2))
```

We generate noisy observations to use for the parameter estimation tasks in this tutorial.
With the [`saveat` argument](https://docs.sciml.ai/latest/basics/common_solver_opts/) we can specify that the solution is stored only at `saveat` time units(default saveat=1 / 50.0).
To make the example more realistic we add random normally distributed noise to the simulation.


```julia
alg1 = NeuralPDE.BNNODE(chainflux1,
    dataset = dataset,
    draw_samples = 1000,
    l2std = [
        0.05,
        0.05,
    ],
    phystd = [
        0.05,
        0.05,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        Normal(4.5,
            5),
        Normal(7,
            2),
        Normal(5,
            2),
        Normal(-4,
            6),
    ],
    n_leapfrog = 30, progress = true)

sol_flux_pestim = solve(prob, alg1)


alg2 = NeuralPDE.BNNODE(chainlux1,
    dataset = dataset,
    draw_samples = 1000,
    l2std = [
        0.05,
        0.05,
    ],
    phystd = [
        0.05,
        0.05,
    ],
    priorsNNw = (0.0,
        3.0),
    param = [
        Normal(4.5,
            5),
        Normal(7,
            2),
        Normal(5,
            2),
        Normal(-4,
            6),
    ],
    n_leapfrog = 30, progress = true)

sol_lux_pestim = solve(prob, alg2)

#testing timepoints must match saveat timepoints of solve() call
t=collect(Float64,prob.tspan[1]:1/50.0:prob.tspan[2])

# plotting solution for x,y(NN approximate by .estimated_nn_params)
plot(t,sol_flux_pestim.ensemblesol[1])
plot!(t,sol_flux_pestim.ensemblesol[2])
sol_flux_pestim.estimated_nn_params

# estimated ODE parameters \alpha, \beta , \delta ,\gamma
sol_flux_pestim.estimated_ode_params

# plotting solution for x,y(NN approximate by .estimated_nn_params)
plot(t,sol_lux_pestim.ensemblesol[1])
plot!(t,sol_lux_pestim.ensemblesol[2])
sol_lux_pestim.estimated_nn_params

# estimated ODE parameters \alpha, \beta , \delta ,\gamma
sol_lux_pestim.estimated_ode_params
```
