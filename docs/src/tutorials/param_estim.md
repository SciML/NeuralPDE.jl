# Optimizing Parameters (Solving Inverse Problems) with Physics-Informed Neural Networks (PINNs)

Consider a Lorenz System,

```math
\begin{align*}
    \frac{\mathrm{d} x}{\mathrm{d}t} &= \sigma (y -x) \, ,\\
    \frac{\mathrm{d} y}{\mathrm{d}t} &= x (\rho - z) - y \, ,\\
    \frac{\mathrm{d} z}{\mathrm{d}t} &= x y - \beta z \, ,\\
\end{align*}
```

with Physics-Informed Neural Networks. Now we would consider the case where we want to optimize the parameters `\sigma`, `\beta`, and `\rho`.

We start by defining the problem,

```@example param_estim
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
      Plots
import ModelingToolkit: Interval, infimum, supremum
@parameters t, σ_, β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]
dt = 0.01
```

And the neural networks as,

```@example param_estim
input_ = length(domains)
n = 8
chain1 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
                   Dense(n, 1))
chain2 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
                   Dense(n, 1))
chain3 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
                   Dense(n, 1))
```

We will add another loss term based on the data that we have to optimize the parameters.

Here we simply calculate the solution of the Lorenz system with [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/#Example-2:-Solving-Systems-of-Equations) based on the adaptivity of the ODE solver. This is used to introduce non-uniformity to the time series.

```@example param_estim
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5(), dt = 0.1)
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]
function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us, ts_]
end
data = getData(sol)

(u_, t_) = data
len = length(data[2])
```

Then we define the additional loss function `additional_loss(phi, θ , p)`, the function has
three arguments:

  - `phi` the trial solution
  - `θ` the parameters of neural networks
  - the hyperparameters `p` .

For a Lux neural network, the composed function will present itself as having θ as a
[`ComponentArray`](https://docs.sciml.ai/ComponentArrays/stable/)
subsets `θ.x`, which can also be dereferenced like `θ[:x]`. Thus, the additional
loss looks like:

```@example param_estim
depvars = [:x, :y, :z]
function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[i](t_, θ[depvars[i]]) .- u_[[i], :]) / len for i in 1:1:3)
end
```

#### Note about Flux

If Flux neural network is used, then the subsetting must be computed manually as `θ`
is simply a vector. This looks like:

```julia
init_params = [Flux.destructure(c)[1] for c in [chain1, chain2, chain3]]
acum = [0; accumulate(+, length.(init_params))]
sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]
(u_, t_) = data
len = length(data[2])

function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[i](t_, θ[sep[i]]) .- u_[[i], :]) / len for i in 1:1:3)
end
```

#### Back to our originally scheduled programming

Then finally defining and optimizing using the `PhysicsInformedNN` interface.

```@example param_estim
discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3],
                                             NeuralPDE.GridTraining(dt), param_estim = true,
                                             additional_loss = additional_loss)
@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t), z(t)], [σ_, ρ, β],
                              defaults = Dict([p .=> 1.0 for p in [σ_, ρ, β]]))
prob = NeuralPDE.discretize(pde_system, discretization)
callback = function (p, l)
    println("Current loss is: $l")
    return false
end
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 5000)
p_ = res.u[(end - 2):end] # p_ = [9.93, 28.002, 2.667]
```

And then finally some analysis by plotting.

```@example param_estim
minimizers = [res.u.depvar[depvars[i]] for i in 1:3]
ts = [infimum(d.domain):(dt / 10):supremum(d.domain) for d in domains][1]
u_predict = [[discretization.phi[i]([t], minimizers[i])[1] for t in ts] for i in 1:3]
plot(sol)
plot!(ts, u_predict, label = ["x(t)" "y(t)" "z(t)"])
```

![Plot_Lorenz](https://user-images.githubusercontent.com/12683885/110944192-2ae05f00-834d-11eb-910b-f5c06d22ec8a.png)
