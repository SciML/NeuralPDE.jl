using Flux, Zygote, StochasticDiffEq
using LinearAlgebra, Statistics
println("NNPDENS_tests")
using Test, NeuralPDE

using Random
Random.seed!(100)

println("one-dimensional heat equation")
x0 = Float32[11.] # initial points
tspan = (0.0f0,5.0f0)
dt = 0.5 # time step
d = 1 # number of dimensions
m = 10 # number of trajectories (batch size)

g(X) = sum(X.^2)   # terminal condition
f(X,u,σᵀ∇u,p,t) = Float32(0.0)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

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
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, pdealg, verbose=true, maxiters=200, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])

error_l2 = sqrt((ans-analytical_ans)^2/ans^2)

println("one-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1


println("high-dimensional heat equation")
d = 50 # number of dimensions
x0 = fill(8.0f0,d)
tspan = (0.0f0,2.0f0)
dt = 0.5
m = 50 # number of trajectories (batch size)

g(X) = sum(X.^2)
f(X,u,σᵀ∇u,p,t) = Float32(0.0)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

hls = 10 + d #hidden layer size
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, pdealg, verbose=true, maxiters=150, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])
error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

println("high-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 1.0

println("Black-Scholes-Barenblatt equation")
d = 30 # number of dimensions
x0 = repeat([1.0f0, 0.5f0], div(d,2))
tspan = (0.0f0,1.0f0)
dt = 0.2
m = 30 # number of trajectories (batch size)

r = 0.05f0
sigma = 0.4f0
f(X,u,σᵀ∇u,p,t) = r * (u - sum(X.*σᵀ∇u))
g(X) = sum(X.^2)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(sigma*X) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

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

ans = solve(prob, pdealg, verbose=true, maxiters=150, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

u_analytical(x, t) = exp((r + sigma^2).*(tspan[end] .- tspan[1])).*sum(x.^2)
analytical_ans = u_analytical(x0, tspan[1])
error_l2 = sqrt((ans .- analytical_ans)^2/ans^2)

println("Black Scholes Barenblatt equation")
# println("numerical ans= ", ans)
# println("analytical ans = " , analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 1.0

# Allen-Cahn Equation
d = 10 # number of dimensions
x0 = fill(0.0f0,d)
tspan = (0.3f0,0.6f0)
dt = 0.015 # time step
m = 20 # number of trajectories (batch size)

g(X) = 1.0 / (2.0 + 0.4*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = u .- u.^3
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

hls = 20 + d #hidden layer size
opt = Flux.ADAM(5^-3)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, pdealg, verbose=true, maxiters=150, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

prob_ans = 0.30879
error_l2 = sqrt((ans - prob_ans)^2/ans^2)

println("Allen-Cahn equation")
# println("numerical = ", ans)
# println("prob_ans = " , prob_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 1.0


# Hamilton Jacobi Bellman Equation
d = 30 # number of dimensions
x0 = fill(0.0f0,d)
tspan = (0.0f0, 1.0f0)
dt = 0.2
m = 30 # number of trajectories (batch size)
λ = 1.0f0
#
g(X) = log(0.5f0 + 0.5f0*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = -λ*sum(σᵀ∇u.^2)
μ_f(X,p,t) = zero(X)  #Vector d x 1 λ
σ_f(X,p,t) = Diagonal(sqrt(2.0f0)*ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

hls = 256 #hidden layer size
opt = Flux.ADAM(0.1)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

@time ans = solve(prob, pdealg, verbose=true, maxiters=150, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-4)

T = tspan[2]
MC = 10^5
W() = randn(d,1)
u_analytical(x, t) = -(1/λ)*log(mean(exp(-λ*g(x .+ sqrt(2.0)*abs.(T-t).*W())) for _ = 1:MC))
analytical_ans = u_analytical(x0, tspan[1])

error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

println("Hamilton Jacobi Bellman Equation")
# println("numerical = ", ans)
# println("analytical = " , analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 1.0


# Nonlinear Black-Scholes Equation with Default Risk
d = 20 # number of dimensions
x0 = fill(100.0f0,d)
tspan = (0.0f0,1.0f0)
dt = 0.125 # time step
m = 20 # number of trajectories (batch size)

g(X) = minimum(X)
δ = 2.0f0/3
R = 0.02f0
f(X,u,σᵀ∇u,p,t) = -(1 - δ)*Q(u)*u - R*u

vh = 50.0f0
vl = 70.0f0
γh = 0.2f0
γl = 0.02f0
function Q(u)
    Q = 0
    if u < vh
        Q = γh
    elseif  u >= vl
        Q = γl
    else  #if  u >= vh && u < vl
        Q = ((γh - γl) / (vh - vl)) * (u - vh) + γh
    end
end

µc = 0.02f0
σc = 0.2f0

μ_f(X,p,t) = µc*X #Vector d x 1
σ_f(X,p,t) = σc*Diagonal(X) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

hls = 256 #hidden layer size
opt = Flux.ADAM(0.008)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

@time ans = solve(prob, pdealg, verbose=true, maxiters=100, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

prob_ans = 57.3
error_l2 = sqrt((ans - prob_ans)^2/ans^2)

println("Nonlinear Black-Scholes Equation with Default Risk")
# println("numerical = ", ans)
# println("prob_ans = " , prob_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 1.0
