using Flux, Test
using NeuralNetDiffEq, LinearAlgebra

# one-dimensional heat equation
x0 = [11.0]  # initial points
t0 = 0     # initial time
T = 5      # terminal time
dt = 0.5   # time step
d = 1      # number of dimensions
m = 50     # number of trajectories (batch size)
grid = (x0, t0, T, dt, d, m)

g(x) = sum(x.^2)   # terminal condition
f(t, x, Y, Z) = 0.0  # function from solved equation
μ(t,x) = 0.0
σ(t,x) = 1.0
prob = (g, f, μ, σ)


hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.005)  #optimizer
#sub-neural network approximating solutions at the desired point
u0(hls, d) = Flux.Chain(Dense(d,hls,relu),
                              Dense(hls,hls,relu),
                              Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u(hls, d) = Flux.Chain(Dense(d,hls,relu),
                          Dense(hls,hls,relu),
                          Dense(hls,d))

# hide_layer_size
neuralNetParam = (hls, opt, u0, σᵀ∇u)

ans = NeuralNetDiffEq.pde_solve(prob, grid, neuralNetParam, verbose = true, abstol=1e-8, maxiters = 300)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, T)
# error = abs(ans - analytical_ans)
error_l2 = sqrt((ans-analytical_ans)^2/ans^2)

# println("one-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
# println("error = ", error)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1


# high-dimensional heat equation
d = 100 # number of dimensions
x0 = fill(8,d)
t0 = 0
T = 2
dt = 0.5
m = 100 # number of trajectories (batch size)
grid = (x0, t0, T, dt, d, m)

g(x) = sum(x.^2)
f(t,x,Y,Z) = 0
μ(t,x) = 0
σ(t,x) = 1
prob = (g, f, μ, σ)


hls = 10 + d #hidden layer size
# hide_layer_size
neuralNetParam = (hls, opt, u0, σᵀ∇u)

ans = NeuralNetDiffEq.pde_solve(prob, grid, neuralNetParam, verbose = true, abstol=1e-8, maxiters = 400)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, T)
# error = abs(ans - analytical_ans)
error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

# println("high-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
# println("error = ", error)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1


#Black-Scholes-Barenblatt equation
d = 100 # number of dimensions
x0 = repeat([1, 0.5], div(d,2))
t0 = 0
T = 1
dt = 0.25
m = 100 # number of trajectories (batch size)
grid = (x0, t0, T, dt, d, m)

r = 0.05
sigma_max = 0.4
f(t, x, Y, Z) = r * (Y .- sum(x.*Z)) # M x 1
g(x) = sum(x.^2)  # M x D
μ(t, x) = 0.0
σ(t, x) = Diagonal(sigma_max*x)
prob = (g, f, μ, σ)

hls  = 10 + d #hide layer size
opt = Flux.ADAM(0.001)
u0(hide_layer_size, d) = Flux.Chain(Dense(d,hls,relu),
                                    Dense(hls,hls,relu),
                                    Dense(hls,hls,relu),
                                    Dense(hls,1))
σᵀ∇u(hide_layer_size, d) = Flux.Chain(Dense(d,hls,relu),
                                       Dense(hls,hls,relu),
                                       Dense(hls,hls,relu),
                                       Dense(hls,d))

neuralNetParam = (hls, opt, u0, σᵀ∇u)

ans = NeuralNetDiffEq.pde_solve(prob, grid, neuralNetParam, verbose = true, abstol=1e-8, maxiters = 250)

u_analytical(x, t) = exp((r + sigma_max^2).*(T .- t)).*sum(x.^2)
analytical_ans = u_analytical(x0, t0)
# error = abs(ans - analytical_ans)
error_l2 = sqrt((ans .- analytical_ans)^2/ans^2)

# println("Black Scholes Barenblatt equation")
# println("numerical ans= ", ans)
# println("analytical ans = " , u_analytical(x0, t0))
# println("error = ", error)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1

#Black–Scholes Equation with Default Risk
#...
#

# Allen-Cahn Equation
# d = 1 # number of dimensions 20
# x0 = fill(0,d)
# t0 = 0
# T = 0.3
# dt = 0.05
# m = 100 # number of trajectories (batch size)
# grid = (x0, t0, T, dt, d, m)
#
# f(t, x, Y, Z) = - Y .+ Y.^3  # M x 1
# g(x) = 1.0 / (2.0 + 0.4*sum(x.^2))
# μ(t, x) = 0 #0
# σ(t, x) = 1 #sigma_max*x
# prob = (g, f, μ, σ)
#
# ans = NeuralNetDiffEq.pde_solve(prob, grid, neuralNetParam, verbose = true, abstol=1e-8, maxiters = 300)
#
# prob_ans = 0.30879
# error = abs(ans - analytical_ans)
#
# println("Allen-Cahn equation")
# println("numerical = ", ans)
# println("analytical = " , prob_ans)
# println("error = ", error)

#Hamilton Jacobi Bellman Equation
# d = 1 # number of dimensions
# x0 = fill(0,d)
# t0 = 0
# T = 1
# dt = 0.1
# m = 100 # number of trajectories (batch size)
# grid = (x0, t0, T, dt, d, m)
#
# f(t, x, Y, Z) = sum(Z.^2) # M x 1
# g(x) = log(0.5 .+ 0.5*sum(x.^2))
# μ(t, x) = 0 #0
# σ(t, x) = sqrt(2) #sigma_max*x
# prob = (g, f, μ, σ)
#
# ans = NeuralNetDiffEq.pde_solve(prob, grid, neuralNetParam, verbose = true, abstol=1e-8, maxiters = 300)

#W =?
# u_analytical(x, t) = log(mean(exp(-g(x + srt(2.0*abs(T-t)*W )))))
# analytical_ans = u_analytical(x0, t0)
# error = abs(ans - analytical_ans)
#
# println("Allen-Cahn equation")
# println("numerical = ", ans)
# println("analytical = " , prob_ans)
# println("error = ", error)
