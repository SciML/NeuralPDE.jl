using  NeuralNetDiffEq, Flux , DifferentialEquations, Optim, Adapt
function f(u, p, W, t)
    2u*sin(W)
end
u0 = 1.00
tspan  = (0.0 , 5.0)
prob = RODEProblem(f,u0,tspan)
dt = 0.1/20f0
t = tspan[1]:dt:tspan[2]
brownian_values = cumsum([0;[sqrt(dt)*randn() for i in 1:length(t)-2]])
W = NoiseGrid(t,brownian_values)
chain = Flux.Chain(Dense(2,5,σ),Dense(5,16 ,σ ) , Dense(16,1))
opt = BFGS()
sol = solve(prob, NeuralNetDiffEq.NNRODE(chain,W,opt), dt=1/20f0,
verbose = true, abstol=1e-10, maxiters = 100)
