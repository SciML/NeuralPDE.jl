using Flux, StochasticDiffEq
using LinearAlgebra, Statistics
using Test, NeuralNetDiffEq

# one-dimensional heat equation
x0 = Float32[11.] # initial points
tspan = (0.0f0,5.0f0)
dt = 0.5 # time step
d = 1 # number of dimensions
m = 50 # number of trajectories (batch size)

g(X) = sum(X.^2)   # terminal condition
f(X,u,σᵀ∇u,p,t) = Float32(0.0)
μ(X,p,t) = zero(X) #Vector d x 1
σ(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.005)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
alg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, alg, verbose=true, maxiters=200, trajectories=m,
                            sde_algorithm=EM(), dt=dt, abstol = 1f-6, reltol = 1f-5)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])

error_l2 = sqrt((ans-analytical_ans)^2/ans^2)

println("one-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1


# high-dimensional heat equation
d = 100 # number of dimensions
x0 = fill(8.0f0,d)
tspan = (0.0f0,2.0f0)
dt = 0.5
m = 100 # number of trajectories (batch size)

g(X) = sum(X.^2)
f(X,u,σᵀ∇u,p,t) = Float32(0.0)
μ(X,p,t) = zero(X) #Vector d x 1
σ(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls = 10 + d #hidden layer size
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
alg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, alg, verbose=true, maxiters=250, trajectories=m,
                            sde_algorithm=EM(), dt=dt, abstol = 1f-6, reltol = 1f-5)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])
error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

println("high-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1


# Black-Scholes-Barenblatt equation
d = 100 # number of dimensions
x0 = repeat([1.0f0, 0.5f0], div(d,2))
tspan = (0.0f0,1.0f0)
dt = 0.25
m = 100 # number of trajectories (batch size)

r = 0.05f0
sigma_max = 0.4f0
f(X,u,σᵀ∇u,p,t) = r * (u .- sum(X.*σᵀ∇u))
g(X) = sum(X.^2)
μ(X,p,t) = zero(X) #Vector d x 1
σ(X,p,t) = Diagonal(sigma_max*X.data) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls  = 10 + d #hide layer size
opt = Flux.ADAM(0.001)
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
alg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, alg, verbose=true, maxiters=150, trajectories=m,
                            sde_algorithm=EM(), dt=dt, abstol = 1f-6, reltol = 1f-5)

u_analytical(x, t) = exp((r + sigma_max^2).*(tspan[end] .- tspan[1])).*sum(x.^2)
analytical_ans = u_analytical(x0, tspan[1])
error_l2 = sqrt((ans .- analytical_ans)^2/ans^2)

println("Black Scholes Barenblatt equation")
# println("numerical ans= ", ans)
# println("analytical ans = " , analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1

# Allen-Cahn Equation
d = 20 # number of dimensions
x0 = fill(0.0f0,d)
tspan = (0.0f0,0.3f0)
dt = 0.015 # time step
m = 100 # number of trajectories (batch size)

g(X) = 1.0 / (2.0 + 0.4*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = u .- u.^3
μ(X,p,t) = zero(X) #Vector d x 1
σ(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls = 10 + d #hidden layer size
opt = Flux.ADAM(5^-4)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
alg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, alg, verbose=true, maxiters=200, trajectories=m,
                            sde_algorithm=EM(), dt=dt, abstol = 1f-6, reltol = 1f-5)

prob_ans = 0.30879
error_l2 = sqrt((ans - prob_ans)^2/ans^2)

println("Allen-Cahn equation")
# println("numerical = ", ans)
# println("prob_ans = " , prob_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1

# Hamilton Jacobi Bellman Equation
d = 100 # number of dimensions
x0 = fill(0.0f0,d)
tspan = (0.0f0, 1.0f0)
dt = 0.2
m = 100 # number of trajectories (batch size)

g(X) = log(0.5f0 + 0.5f0*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = sum(σᵀ∇u.^2)
μ(X,p,t) = zero(X) #Vector d x 1
σ(X,p,t) = Diagonal(sqrt(2.0f0)*ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.001)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
alg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, alg, verbose=true, maxiters=100, trajectories=m,
                            sde_algorithm=EM(), dt=dt, abstol = 1f-6, reltol = 1f-5)

ts = tspan[1]:dt:tspan[2]
T = tspan[2]
NC = length(ts)
MC = 10^5
W() = randn(d,NC)
u_analytical(x, ts) = -log(mean(exp(-g(x .+ sqrt(2.0)*abs.(T.-ts').*W())) for _ = 1:MC))
analytical_ans = u_analytical(x0, ts)

error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

println("Hamilton Jacobi Bellman Equation")
# println("numerical = ", ans)
# println("analytical = " , analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.2
