using Test, Flux, StochasticDiffEq
println("Kolmogorov Tests")
using NeuralPDE
using Distributions

using Random
Random.seed!(100)

#Using SDEProblem for the Algorithm.
# For a diract delta take u0 = Normal(0 , sigma) where sigma --> 0
u0 = Normal(1.00, 1.00)
xspan = (-2.0, 6.0)
tspan = (0.0, 1.0)
g(u, p, t) = 2.00
f(u, p, t) = -2.00
d = 1
sdealg = EM()
prob = SDEProblem(f, g, u0, (0.0, 1.0); xspan = xspan, d = d)
opt = Flux.ADAM(0.01)
m = Chain(Dense(1, 5, elu), Dense(5, 5, elu), Dense(5, 5, elu), Dense(5, 1))
ensemblealg = EnsembleThreads()
sol = solve(prob, NNKolmogorov(m, opt, sdealg, ensemblealg), verbose = true, dt = 0.01,
            abstol = 1e-10, dx = 0.0001, trajectories = 100000, maxiters = 500)
# using Plots
#

# x_val = collect(xs)
# x_val= reshape(x_val , 1 , size(x_val)[1])
# y_val = m(x_val)
# y_val = reshape(y_val , 800001 , 1)
# x_val = collect(xs)
# plot(x_val , y_val,linewidth=3,title="Solution to the linear ODE with a thick line",
#      xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!")
# # plot(x_val , y_val)
# plot!(x_val , analytical(x_val),linewidth=3,title="Solution to the linear ODE with a thick line",
#      xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!")
#

## The solution is obtained taking the Fourier Transform.
analytical(xi) = pdf.(Normal(3, sqrt(1.0 + 5.00)), xi)
##Validation
xs = -5:0.00001:5
x_1 = rand(xs, 1, 1000)
err_l2 = Flux.mse(analytical(x_1), m(x_1))
@test err_l2 < 0.01
##

##Using the KolmogorovPDEProblem.
function phi(xi)
    y = Float64[]
    for x in xi
        y = push!(y, 1.77 * x - 0.015 * x^3)
    end
    y = reshape(y, size(xi)[1], size(xi)[2])
    return y
end

xspan2 = (-6.0, 6.0)
tspan2 = (0.0, 1.0)
#f = mu and g  = sigma
g2(u, p, t) = 0.5 * u
f2(u, p, t) = 0.5 * 0.25 * u
d2 = 1
sdealg2 = EM()
prob2 = KolmogorovPDEProblem(f2, g2, phi, xspan2, tspan2, d2)
opt2 = Flux.ADAM(0.01)
m2 = Chain(Dense(1, 16, elu), Dense(16, 32, elu), Dense(32, 16, elu), Dense(16, 1))
sol = solve(prob2, NeuralPDE.NNKolmogorov(m2, opt2, sdealg2, ensemblealg), verbose = true,
            dt = 0.01,
            dx = 0.0001, trajectories = 1000, abstol = 1e-6, maxiters = 300)

function analytical2(xi)
    y = Float64[]
    a = 1.77 * exp(0.5 * (0.5)^2 * 1.0)
    b = -0.015 * exp(0.5 * (0.5 * 3)^2 * 1.0)
    for x in xi
        y = push!(y, a * x + b * x^3)
    end
    y = reshape(y, size(xi)[1], size(xi)[2])
    return y
end
xs2 = -5.00:0.01:5.00
x_val2 = rand(xs2, d2, 50)
errorl2 = Flux.mse(analytical2(x_val2), m2(x_val2))
println("error_l2 = ", errorl2, "\n")
@test errorl2 < 0.4

##Non-Diagonal Test
f_noise = (du, u, p, t) -> du .= 1.01u
g_noise = function (du, u, p, t)
    du[1, 1] = 0.3u[1]
    du[1, 2] = 0.6u[1]
    du[1, 3] = 0.9u[1]
    du[1, 4] = 0.12u[2]
    du[2, 1] = 1.2u[1]
    du[2, 2] = 0.2u[2]
    du[2, 3] = 0.3u[2]
    du[2, 4] = 1.8u[2]
end
Σ = [1.0 0.3; 0.3 1.0]
uo3 = MvNormal([0.0; 0.0], Σ)
sdealg3 = EM()
xspan3 = (-10.0, 10.0)
tspan3 = (0.0, 1.0)
d3 = 2
prob = SDEProblem(f_noise, g_noise, uo3, (0.0, 1.0); xspan = xspan3, d = d3,
                  noise_rate_prototype = zeros(2, 4))
opt = Flux.ADAM(0.01)
m3 = Chain(Dense(d3, 32, elu), Dense(32, 64, elu), Dense(64, 1))
sol3 = solve(prob, NeuralPDE.NNKolmogorov(m3, opt, sdealg3, EnsembleThreads()),
             verbose = true, dt = 0.001,
             abstol = 1e-6, dx = 0.001, trajectories = 1000, maxiters = 200)
println("Non-Diagonal test working.")
