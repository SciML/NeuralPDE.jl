# Linear parabolic system of PDEs

We can use NeuralPDE to solve the linear parabolic system of PDEs:

```math
\begin{aligned}
\frac{\partial u}{\partial t} &= a * \frac{\partial^2 u}{\partial x^2} + b_1 u + c_1 w \\
\frac{\partial w}{\partial t} &= a * \frac{\partial^2 w}{\partial x^2} + b_2 u + c_2 w \\
\end{aligned}
```

with initial and boundary conditions:

```math
\begin{aligned}
u(0, x) &= \frac{b_1 - \lambda_2}{b_2 (\lambda_1 - \lambda_2)} \cdot \cos(\frac{x}{a}) -  \frac{b_1 - \lambda_1}{b_2 (\lambda_1 - \lambda_2)} \cdot \cos(\frac{x}{a}) \\
w(0, x) &= 0 \\
u(t, 0) &= \frac{b_1 - \lambda_2}{b_2 (\lambda_1 - \lambda_2)} \cdot e^{\lambda_1t} -  \frac{b_1 - \lambda_1}{b_2 (\lambda_1 - \lambda_2)} \cdot e^{\lambda_2t} \\
w(t, 0) &= \frac{e^{\lambda_1}-e^{\lambda_2}}{\lambda_1 - \lambda_2} \\
u(t, 1) &= \frac{b_1 - \lambda_2}{b_2 (\lambda_1 - \lambda_2)} \cdot e^{\lambda_1t} \cdot \cos(\frac{x}{a}) -  \frac{b_1 - \lambda_1}{b_2 (\lambda_1 - \lambda_2)} \cdot e^{\lambda_2t} * \cos(\frac{x}{a}) \\
w(t, 1) &= \frac{e^{\lambda_1} \cos(\frac{x}{a})-e^{\lambda_2} \cos(\frac{x}{a})}{\lambda_1 - \lambda_2}
\end{aligned}
```

with a physics-informed neural network.

```@example linear_parabolic
using NeuralPDE, Lux, OptimizationOptimisers
using Plots
using DomainSets: Interval

@parameters t, x
@variables u(..), w(..)
Dxx = Differential(x)^2
Dt = Differential(t)

# Constants
a = 1
b1 = 4
b2 = 2
c1 = 3
c2 = 1
λ1 = (b1 + c2 + sqrt((b1 + c2)^2 + 4 * (b1 * c2 - b2 * c1))) / 2
λ2 = (b1 + c2 - sqrt((b1 + c2)^2 + 4 * (b1 * c2 - b2 * c1))) / 2

# Analytic solution
θ(t, x) = exp(-t) * cos(x / a)
function u_analytic(t, x)
    (b1 - λ2) / (b2 * (λ1 - λ2)) * exp(λ1 * t) * θ(t, x) -
    (b1 - λ1) / (b2 * (λ1 - λ2)) * exp(λ2 * t) * θ(t, x)
end
w_analytic(t, x) = 1 / (λ1 - λ2) * (exp(λ1 * t) * θ(t, x) - exp(λ2 * t) * θ(t, x))

# Second-order constant-coefficient linear parabolic system
eqs = [Dt(u(t, x)) ~ a * Dxx(u(t, x)) + b1 * u(t, x) + c1 * w(t, x),
    Dt(w(t, x)) ~ a * Dxx(w(t, x)) + b2 * u(t, x) + c2 * w(t, x)]

# Boundary conditions
bcs = [u(0, x) ~ u_analytic(0, x),
    w(0, x) ~ w_analytic(0, x),
    u(t, 0) ~ u_analytic(t, 0),
    w(t, 0) ~ w_analytic(t, 0),
    u(t, 1) ~ u_analytic(t, 1),
    w(t, 1) ~ w_analytic(t, 1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

# Neural network
n = 15
chain = [Chain(Dense(2, n, σ), Dense(n, n, σ), Dense(n, 1)) for _ in 1:2]

strategy = QuasiRandomTraining(500; bcs_points = 100)
discretization = PhysicsInformedNN(chain, strategy)

@named pdesystem = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), w(t, x)])
prob = discretize(pdesystem, discretization)

callback = function (p, l)
    p.iter % 500 == 0 && println("iter: ", p.iter, " loss: ", l)
    return false
end

sol = solve(prob, Adam(1e-2); maxiters = 2000, callback)

# Analysis
ts = xs = 0:0.01:1
dvs = [u(t, x), w(t, x)]

analytic_sol_func(t, x) = [u_analytic(t, x), w_analytic(t, x)]
u_real = [[analytic_sol_func(t, x)[i] for t in ts, x in xs] for i in 1:2]
u_predict = [sol(ts, xs; dv = dvs[i]) for i in 1:2]
diff_u = [abs.(u_real[i] .- u_predict[i]) for i in 1:2]
ps = []
for i in 1:2
    p1 = plot(ts, xs, u_real[i]', linetype = :contourf, title = "u$i, analytic")
    p2 = plot(ts, xs, u_predict[i]', linetype = :contourf, title = "predict")
    p3 = plot(ts, xs, diff_u[i]', linetype = :contourf, title = "error")
    push!(ps, plot(p1, p2, p3))
end
```

```@example linear_parabolic
ps[1]
```

```@example linear_parabolic
ps[2]
```
