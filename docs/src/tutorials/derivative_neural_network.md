# The Derivative neural network approximation

The accuracy and stability of numerical derivative decreases with each successive order.
The accuracy of the entire solution is determined by the worst accuracy of one of the variables,
in our case, the highest degree of the derivative. Meanwhile, the computational cost of automatic
differentiation for higher orders grows as O(n^d), making even numerical differentiation
much more efficient! Given these two bad choices, there exists an alternative which can improve
training speed and accuracy: using a system to represent the derivatives directly.

## Demonstration

Take the PDE system:

```math
\begin{align*}
∂_t^2 u_1(t, x) & = ∂_x^2 u_1(t, x) + u_3(t, x) \, \sin(\pi x) \, ,\\
∂_t^2 u_2(t, x) & = ∂_x^2 u_2(t, x) + u_3(t, x) \, \cos(\pi x) \, ,\\
0 & = u_1(t, x) \sin(\pi x) + u_2(t, x) \cos(\pi x) - e^{-t} \, ,
\end{align*}
```

with the initial conditions:

```math
\begin{align*}
u_1(0, x) & = \sin(\pi x) \, ,\\
∂_t u_1(0, x) & = - \sin(\pi x) \, ,\\
u_2(0, x) & = \cos(\pi x) \, ,\\
∂_t u_2(0, x) & = - \cos(\pi x) \, ,
\end{align*}
```

and the boundary conditions:

```math
\begin{align*}
u_1(t, 0) & = u_1(t, 1) = 0 \, ,\\
u_2(t, 0) & = - u_2(t, 1) = e^{-t} \, ,
\end{align*}
```

This is the same system as the [system of equations example](@ref systems)

The derivative neural network approximation is such an approach that using lower-order numeric
derivatives and estimates higher-order derivatives with a neural network, so that allows an
increase in the marginal precision for all optimization. Since `u3` is only in the first and
second equations, its accuracy during training is determined by the accuracy of the
second numerical derivative `u3(t,x) ~ (Dtt(u1(t,x)) -Dxx(u1(t,x))) / sin(pi*x)`.

We approximate the derivative of the neural network with another neural network
`Dt(u1(t,x)) ~ Dtu1(t,x)` and train it along with other equations, and thus we avoid
using the second numeric derivative `Dt(Dtu1(t,x))`.

```@example derivativenn
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers,
      OptimizationOptimJL, LineSearches, Plots
using ModelingToolkit: Interval, infimum, supremum

@parameters t, x
Dt = Differential(t)
Dx = Differential(x)
@variables u1(..), u2(..), u3(..)
@variables Dxu1(..) Dtu1(..) Dxu2(..) Dtu2(..)

eqs_ = [
    Dt(Dtu1(t, x)) ~ Dx(Dxu1(t, x)) + u3(t, x) * sinpi(x),
    Dt(Dtu2(t, x)) ~ Dx(Dxu2(t, x)) + u3(t, x) * cospi(x),
    exp(-t) ~ u1(t, x) * sinpi(x) + u2(t, x) * cospi(x)
]

bcs_ = [
    u1(0.0, x) ~ sinpi(x),
    u2(0.0, x) ~ cospi(x),
    Dt(u1(0, x)) ~ -sinpi(x),
    Dt(u2(0, x)) ~ -cospi(x),
    u1(t, 0.0) ~ 0.0,
    u2(t, 0.0) ~ exp(-t),
    u1(t, 1.0) ~ 0.0,
    u2(t, 1.0) ~ -exp(-t)
]

der_ = [
    Dt(u1(t, x)) ~ Dtu1(t, x),
    Dt(u2(t, x)) ~ Dtu2(t, x),
    Dx(u1(t, x)) ~ Dxu1(t, x),
    Dx(u2(t, x)) ~ Dxu2(t, x)
]

bcs__ = [bcs_; der_]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

input_ = length(domains)
n = 15
chain = [@closure Chain(Dense(input_, n, σ), Dense(n, n, σ), Dense(n, 1)) for _ in 1:7]

training_strategy = StochasticTraining(128)
discretization = PhysicsInformedNN(chain, training_strategy)

vars = [u1(t, x), u2(t, x), u3(t, x), Dxu1(t, x), Dtu1(t, x), Dxu2(t, x), Dtu2(t, x)]
@named pdesystem = PDESystem(eqs_, bcs__, domains, [t, x], vars)
prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions[1:7]
approx_derivative_loss_functions = sym_prob.loss_functions.bc_loss_functions[9:end]

callback = @closure function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    println("der_losses: ", map(l_ -> l_(p.u), approx_derivative_loss_functions))
    return false
end

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 2000, callback)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, LBFGS(linesearch = BackTracking()); maxiters = 200, callback)

phi = discretization.phi
```

And some analysis:

```@example derivativenn
using Plots

ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
minimizers_ = [@closure res.u.depvar[sym_prob.depvars[i]] for i in 1:length(chain)]

u1_real(t, x) = exp(-t) * sinpi(x)
u2_real(t, x) = exp(-t) * cospi(x)
u3_real(t, x) = (1 + pi^2) * exp(-t)
Dxu1_real(t, x) = pi * exp(-t) * cospi(x)
Dtu1_real(t, x) = -exp(-t) * sinpi(x)
Dxu2_real(t, x) = -pi * exp(-t) * sinpi(x)
Dtu2_real(t, x) = -exp(-t) * cospi(x)

function analytic_sol_func_all(t, x)
    [u1_real(t, x), u2_real(t, x), u3_real(t, x),
        Dxu1_real(t, x), Dtu1_real(t, x), Dxu2_real(t, x), Dtu2_real(t, x)]
end

u_real = [[analytic_sol_func_all(t, x)[i] for t in ts for x in xs] for i in 1:7]
u_predict = [[phi[i]([t, x], minimizers_[i])[1] for t in ts for x in xs] for i in 1:7]
diff_u = [abs.(u_real[i] .- u_predict[i]) for i in 1:7]
ps = []
titles = ["u1", "u2", "u3", "Dtu1", "Dtu2", "Dxu1", "Dxu2"]
for i in 1:7
    p1 = plot(ts, xs, u_real[i], linetype = :contourf, title = "$(titles[i]), analytic")
    p2 = plot(ts, xs, u_predict[i], linetype = :contourf, title = "predict")
    p3 = plot(ts, xs, diff_u[i], linetype = :contourf, title = "error")
    push!(ps, plot(p1, p2, p3))
end
```

```@example derivativenn
ps[1]
```

```@example derivativenn
ps[2]
```

```@example derivativenn
ps[3]
```

```@example derivativenn
ps[4]
```

```@example derivativenn
ps[5]
```

```@example derivativenn
ps[6]
```

```@example derivativenn
ps[7]
```
