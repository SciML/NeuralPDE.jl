using Test, Flux
using NeuralPDE

d = 1
m = Chain(Dense(3 , 64, elu),Dense(64, 64, elu) , Dense(64 , 5 , elu) , Dense(5 , 1))
ensemblealg = EnsembleThreads()
γ_mu_prototype = nothing
γ_sigma_prototype = zeros(d , d , 1)
γ_phi_prototype = nothing

sdealg = EM()
tspan = [0.00 , 1.00]
trajectories = 100000
function phi(x)
  x.^2
end
xspan = [0.00 , 5.00]
yspan = [0.00 , 1.00]
dt = 0.01
dx = 0.01
dy = 0.01
prob = ParamKolmogorovPDEProblem(f , g , u0 , (0.0 , 1.0) ; xspan = xspan , d = d)
sol = solve(prob, NNKolmogorov(m,opt , sdealg,ensemblealg) , verbose = true, dt = 0.01,
            abstol=1e-10, dx = 0.0001 , trajectories = 100000 ,  maxiters = 500)
