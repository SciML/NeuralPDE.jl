# Neural Network Solvers for Kolmogorov Backwards Equations

A Kolmogorov PDE is of the form :

![](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/KolmogorovPDE.png)

Considering S be a solution process to the SDE:

![](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/StochasticP.png)

then the solution to the Kolmogorov PDE is given as:

![](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/Solution.png)

A Kolmogorov PDE Problem can be defined using a `SDEProblem`:

```julia
SDEProblem(μ,σ,u0,tspan,xspan,d)
```

Here `u0` is the initial distribution of x. Here we define `u(0,x)` as the probability density function of `u0`.`μ` and `σ` are obtained from the SDE for the stochastic process above. `d` represents the dimenstions of `x`.
`u0` can be defined using `Distributions.jl`.

Another was of defining a KolmogorovPDE is using the `KolmogorovPDEProblem`.

```julia
KolmogorovPDEProblem(μ,σ,phi,tspan,xspan,d)
```

Here `phi` is the initial condition on u(t,x) when t = 0. `μ` and `σ` are obtained from the SDE for the stochastic process above. `d` represents the dimenstions of `x`.

To solve this problem use,

- `NNKolmogorov(chain, opt , sdealg)`: Uses a neural network to realise a regression function which is the solution for the linear Kolmogorov Equation.

Here, `chain` is a Flux.jl chain with `d` dimensional input and 1 dimensional output.`opt` is a Flux.jl optimizer. And `sdealg` is a high-order algorithm to calculate the solution for the SDE, which is used to define the learning data for the problem. Its default value is the classic Euler-Maruyama algorithm.

## Using GPU for Kolmogorov Equations

For running Kolmogorov Equations on a GPU there are somethings that are needed to be taken care of :

Convert the model parameters to `CuArrays` using the `fmap` function given by Flux.jl
```julia
m = Chain(Dense(1, 64, σ), Dense(64, 64, σ) , Dense(5, 2))
m = fmap(cu, m)
```
Unlike other solver we need to specify explicitly to the solver to run on the GPU. This can be done by passing the `use_gpu = true`  into the solver.

```julia
solve(prob, NeuralNetDiffEq.NNKolmogorov(m, opt, sdealg, ensemblealg), use_gpu = true,  verbose = true, dt = dt, dx = dx , trajectories = trajectories , abstol=1e-6, maxiters = 1000)
```
