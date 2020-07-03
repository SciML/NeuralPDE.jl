# Optimal Stopping Problem Solver
Here we will aim to solve an optimal stopping problem using `NNStopping` algorithm.

Let us consider a standard American options. Unlike European options , American options can be excersiced before their maturity and thus the problem reduces to finding an optimal stopping time.
We will take the case of an American max put option with strike price `K` , constant volatility `β` and risk-free rate `r` . Intitial stock price S<sub>0</sub> = 80.00 , the maturity `T` and number of steps as `N`
```julia
d = 1 #Dimensions of initial stock price
r = 0.04f0
beta = 0.2f0
K = 100.00
T = 1.0
u0 = fill(80.00 , d , 1)
f(du,u,p,t) = (du .= r*u)
sigma(du,u,p,t)  = (du .= Diagonal(beta*u))
tspan = (0.0 , T)
N = 50
dt = tspan[2]/(N - 1)
```
The final part is the payoff function :
<p align="center">
  <img src="https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/payoff_function.png">
</p>
The discounted payoff function is :

```julia
function g(t , x)
  return exp(-r*t)*(max(K -  maximum(x)  , 0))
end
```
Now in order to define an optimal stopping problem we will use a `SDEProblem` and pass the discounted payoff function `g`as an `kwarg`.
```julia
prob  = SDEProblem(f , sigma , u0 , tspan ; g = g)
```
And finally lets build our neural network model using Flux.jl. Note that the final layer should be the softmax (Flux.softmax)  function as we need the sum of probabilities at all stopping times to be 1. And then add an optimiser function.
```julia
m = Chain(Dense(d , 5, tanh), Dense(5, 16 , tanh)  , Dense(16 , N ), softmax)
opt = Flux.ADAM(0.1)
```
We add algorithms to solve the SDE and the Ensemble.
```julia
sdealg = EM()
ensemblealg = EnsembleThreads()
```
And finally we call the solve function.
```julia
sol = solve(prob, NeuralNetDiffEq.NNStopping( m, opt , sdealg , ensemblealg), verbose = true, dt = dt,
            abstol=1e-6, maxiters = 20 , trajectories = 200)

```
