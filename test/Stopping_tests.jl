using Test, Flux , StochasticDiffEq , LinearAlgebra
println("Optimal Stopping Time Test")
using NeuralNetDiffEq
d = 1
r = 0.04f0
beta = 0.2f0
T = 1
u0 = fill(80.00 , d , 1)
sdealg = EM()
ensemblealg = EnsembleThreads()
f(du,u,p,t) = (du .= r*u)
sigma(du,u,p,t)  = (du .= Diagonal(beta*u))
tspan = (0.0 , 1.0)
N = 50
dt = tspan[2]/49
K = 100.00
function g(t , x)
  return exp(-r*t)*(max(K -  maximum(x)  , 0))
end

prob  = SDEProblem(f , sigma , u0 , tspan ; g = g)
opt = Flux.ADAM(0.1)
m = Chain(Dense(d , 5, tanh), Dense(5, 16 , tanh)  , Dense(16 , N ), softmax)
sol = solve(prob, NeuralNetDiffEq.NNStopping( m, opt , sdealg , ensemblealg), verbose = true, dt = dt,
            abstol=1e-6, maxiters = 20 , trajectories = 200)

##Analytical Binomial Tree approach for American Options
function BinomialTreeAM1D(S0 , N , r , beta)
    V = zeros(N+1)
    dT = T/N
    u = exp(beta*sqrt(dT))
    d = 1/u
    S_T = [S0*(u^j)* (d^(N-j)) for j in 0:N]
    a = exp(r*dT)
    p = (a - d)/(u - d)
    q = 1.0 - p
    V = [max(K - x , 0) for x in S_T]
    for i in N-1:-1:0
      V[1:end-1] = exp(-r*dT).*(p*V[2:end] + q*V[1:end-1])
      S_T = S_T*u
      V = [max(K - S_T[i] , V[i]) for i in 1:size(S_T)[1]]
    end
    return V[1]
end
real_sol = BinomialTreeAM1D(u0[1] , N , r , beta)
error = abs(sol - real_sol)
@test error < 0.5
