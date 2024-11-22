# Using `ahmc_bayesian_pinn_pde` with the `BayesianPINN` Discretizer for the Kuramoto–Sivashinsky equation

Consider the Kuramoto–Sivashinsky equation:

```math
∂_t u(x, t) + u(x, t) ∂_x u(x, t) + \alpha ∂^2_x u(x, t) + \beta ∂^3_x u(x, t) + \gamma ∂^4_x u(x, t) =  0 \, ,
```

where $\alpha = \gamma = 1$ and $\beta = 4$. The exact solution is:

```math
u_e(x, t) = 11 + 15 \tanh \theta - 15 \tanh^2 \theta - 15 \tanh^3 \theta \, ,
```

where $\theta = t - x/2$ and with initial and boundary conditions:

```math
\begin{align*}
    u(  x, 0) &=     u_e(  x, 0) \, ,\\
    u( 10, t) &=     u_e( 10, t) \, ,\\
    u(-10, t) &=     u_e(-10, t) \, ,\\
∂_x u( 10, t) &= ∂_x u_e( 10, t) \, ,\\
∂_x u(-10, t) &= ∂_x u_e(-10, t) \, .
\end{align*}
```

With Bayesian Physics-Informed Neural Networks, here is an example of using `BayesianPINN` discretization with `ahmc_bayesian_pinn_pde` :

```@example low_level_2
using NeuralPDE, Lux, ModelingToolkit, LinearAlgebra, AdvancedHMC
import ModelingToolkit: Interval, infimum, supremum, Distributions
using Plots, MonteCarloMeasurements

@parameters x, t, α
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dx2 = Differential(x)^2
Dx3 = Differential(x)^3
Dx4 = Differential(x)^4

# α = 1
β = 4
γ = 1
eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) + α * Dx2(u(x, t)) + β * Dx3(u(x, t)) + γ * Dx4(u(x, t)) ~ 0

u_analytic(x, t; z = -x / 2 + t) = 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
du(x, t; z = -x / 2 + t) = 15 / 2 * (tanh(z) + 1) * (3 * tanh(z) - 1) * sech(z)^2

bcs = [u(x, 0) ~ u_analytic(x, 0),
    u(-10, t) ~ u_analytic(-10, t),
    u(10, t) ~ u_analytic(10, t),
    Dx(u(-10, t)) ~ du(-10, t),
    Dx(u(10, t)) ~ du(10, t)]

# Space and time domains
domains = [x ∈ Interval(-10.0, 10.0),
    t ∈ Interval(0.0, 1.0)]

# Discretization
dx = 0.4;
dt = 0.2;

# Function to compute analytical solution at a specific point (x, t)
function u_analytic_point(x, t)
    z = -x / 2 + t
    return 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
end

# Function to generate the dataset matrix
function generate_dataset_matrix(domains, dx, dt)
    x_values = -10:dx:10
    t_values = 0.0:dt:1.0

    dataset = []

    for t in t_values
        for x in x_values
            u_value = u_analytic_point(x, t)
            push!(dataset, [u_value, x, t])
        end
    end

    return vcat([data' for data in dataset]...)
end

datasetpde = [generate_dataset_matrix(domains, dx, dt)]

# noise to dataset
noisydataset = deepcopy(datasetpde)
noisydataset[1][:, 1] = noisydataset[1][:, 1] .+
                        randn(size(noisydataset[1][:, 1])) .* 5 / 100 .*
                        noisydataset[1][:, 1]
```

Plotting dataset, added noise is set at 5%.

```@example low_level_2
plot(datasetpde[1][:, 2], datasetpde[1][:, 1], title = "Dataset from Analytical Solution")
plot!(noisydataset[1][:, 2], noisydataset[1][:, 1])
```

```@example low_level_2
# Neural network
chain = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))

discretization = NeuralPDE.BayesianPINN([chain],
    GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset, nothing])

@named pde_system = PDESystem(eq,
    bcs,
    domains,
    [x, t],
    [u(x, t)],
    [α],
    defaults = Dict([α => 0.5]))

sol1 = ahmc_bayesian_pinn_pde(pde_system,
    discretization;
    draw_samples = 100, Kernel = AdvancedHMC.NUTS(0.8),
    bcstd = [0.2, 0.2, 0.2, 0.2, 0.2],
    phystd = [1.0], l2std = [0.05], param = [Distributions.LogNormal(0.5, 2)],
    priorsNNw = (0.0, 10.0),
    saveats = [1 / 100.0, 1 / 100.0], progress = true)
```

And some analysis:

```@example low_level_2
phi = discretization.phi[1]
xs, ts = [infimum(d.domain):dx:supremum(d.domain)
          for (d, dx) in zip(domains, [dx / 10, dt])]
u_predict = [[first(pmean(phi([x, t], sol1.estimated_nn_params[1]))) for x in xs]
             for t in ts]
u_real = [[u_analytic(x, t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x, t) - first(pmean(phi([x, t], sol1.estimated_nn_params[1]))))
           for x in xs]
          for t in ts]

p1 = plot(xs, u_predict, title = "predict")
p2 = plot(xs, u_real, title = "analytic")
p3 = plot(xs, diff_u, title = "error")
plot(p1, p2, p3)
```
