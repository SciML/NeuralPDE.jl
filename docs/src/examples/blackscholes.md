# Solving the 100 dimensional Black-Scholes-Barenblatt Equation

Black Scholes equation is a model for stock option price.
In 1973, Black and Scholes transformed their formula on option pricing and corporate liabilities into a PDE model, which is widely used in financing engineering for computing the option price over time. [1.]
In this example we will solve a Black-Scholes-Barenblatt equation of 100 dimensions.
The Black-Scholes-Barenblatt equation is a nonlinear extension to the Black-Scholes
equation which models uncertain volatility and interest rates derived from the
Black-Scholes equation. This model results in a nonlinear PDE whose dimension
is the number of assets in the portfolio.

To solve it using the `TerminalPDEProblem`, we write:

```julia
d = 100 # number of dimensions
X0 = repeat([1.0f0, 0.5f0], div(d,2)) # initial value of stochastic state
tspan = (0.0f0,1.0f0)
r = 0.05f0
sigma = 0.4f0
f(X,u,σᵀ∇u,p,t) = r * (u - sum(X.*σᵀ∇u))
g(X) = sum(X.^2)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(sigma*X) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, X0, tspan)
```

As described in the API docs, we now need to define our `NNPDENS` algorithm
by giving it the Flux.jl chains we want it to use for the neural networks.
`u0` needs to be a `d` dimensional -> 1 dimensional chain, while `σᵀ∇u`
needs to be `d+1` dimensional to `d` dimensions. Thus we define the following:

```julia
hls  = 10 + d #hide layer size
opt = Flux.ADAM(0.001)
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)
```

And now we solve the PDE. Here we say we want to solve the underlying neural
SDE using the Euler-Maruyama SDE solver with our chosen `dt=0.2`, do at most
150 iterations of the optimizer, 100 SDE solves per loss evaluation (for averaging),
and stop if the loss ever goes below `1f-6`.

```julia
ans = solve(prob, pdealg, verbose=true, maxiters=150, trajectories=100,
                            alg=EM(), dt=0.2, pabstol = 1f-6)
```

## Reference

1. Shinde, A. S., and K. C. Takale. "Study of Black-Scholes model and its applications." Procedia Engineering 38 (2012): 270-279.
