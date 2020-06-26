# Solving ODEs with Neural Networks

For ODEs, [see the DifferentialEquations.jl documentation](http://docs.juliadiffeq.org/dev/solvers/ode_solve#NeuralNetDiffEq.jl-1)
for the `nnode(chain,opt=ADAM(0.1))` algorithm, which takes in a Flux.jl chain
and optimizer to solve an ODE. This method is not particularly efficient, but
is parallel. It is based on the work of:

[Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.](https://arxiv.org/pdf/physics/9705023.pdf)

## Solving Kolmogorov Equations with Neural Networks

A Kolmogorov PDE is of the form :

![KPDE](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/KolmogorovPDE.png)

Considering S be a solution process to the SDE:

![StochasticP](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/StochasticP.png)

then the solution to the Kolmogorov PDE is given as:

![Solution](https://raw.githubusercontent.com/ashutosh-b-b/Kolmogorv-Equations-Notebook/master/KolmogorovPDEImages/Solution.png)

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
