# Optimal Stopping Times of American Options

Here, we will aim to solve an optimal stopping problem using the `NNStopping` algorithm.

Let us consider standard American options. Unlike European options, American options can be exercized before their maturity and thus the problem reduces to finding an optimal stopping time.

As stated above, since we can execute the option at any optimal time before the maturity of the option, the standard Black-Scholes model gets modified to:

```math
  \frac{∂V}{∂t} + rS\frac{∂V}{∂S} + \frac{1}{2}{\σ^2}{S^2}\frac{∂^2 V}{\∂S^2} -rV ≤ 0
```
The stock price will follow a standard geometric brownian motion given by:

```math
  dS_t = rS_tdt + σS_tdW_t
```
And thus our final aim will be to calculate:
<img src="https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/Price%20of%20American%20Option.png">

We will be using a `SDEProblem` to denote a problem of this type. We can define this as a `SDEProblem` and add a terminal condition `g` in order to price the American Options.


We will take the case of an American max put option with strike price `K`, constant volatility `β`, a risk-free rate `r`, the initial stock price `u0 = 80.00`, the maturity `T`, and number of steps `N`. The forcing function `f` and noise function `sigma` are defined for the type of model. [See StochasticDiffEq documentation.](https://diffeq.sciml.ai/v6.12/tutorials/sde_example/#Example-1:-Scalar-SDEs-1)
```julia
d = 1 #Dimensions of initial stock price
r = 0.04f0
beta = 0.2f0
K = 100.00
T = 1.0
u0 = fill(80.00 , d , 1) #Initial Stock Price
#Defining the drift (f) and diffusion(sigma)
f(du,u,p,t) = (du .= r*u)
sigma(du,u,p,t)  = (du .= Diagonal(beta*u))

tspan = (0.0 , T)
N = 50
dt = tspan[2]/(N - 1)
```
The final part is the payoff function:

  <img src="https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/payoff_function.png">

The discounted payoff function is:

```julia
function g(t , x)
  return exp(-r*t)*(max(K -  maximum(x)  , 0))
end
```
Now, in order to define an optimal stopping problem, we will use the `SDEProblem` and pass the discounted payoff function `g` as an `kwarg`.
```julia
prob  = SDEProblem(f , sigma , u0 , tspan ; g = g)
```
Finally, let's build our neural network model using Flux.jl. Note that the final layer should be the softmax (Flux.softmax) function as we need the sum of probabilities at all stopping times to be 1. And then add an optimizer function.
```julia
m = Chain(Dense(d , 5, tanh), Dense(5, 16 , tanh)  , Dense(16 , N ), softmax)
opt = Flux.ADAM(0.1)
```
We add algorithms to solve the SDE and the Ensemble. These are the algorithms required to solve the `SDEProblem` (we use the Euler-Maruyama algorithm in this case) and the `EnsembleProblem` to run multiple simulations. [See Ensemble Algorithms.](https://diffeq.sciml.ai/stable/features/ensemble/#EnsembleAlgorithms-1)

```julia
sdealg = EM()
ensemblealg = EnsembleThreads()
```

Finally, we call the solve function:
```julia
sol = solve(prob, NeuralPDE.NNStopping( m, opt , sdealg , ensemblealg), verbose = true, dt = dt,
            abstol=1e-6, maxiters = 20 , trajectories = 200)

```
