using Test, Flux, NeuralNetDiffEq , StochasticDiffEq
using DiffEqDevTools
using Distributions

u0 = 0
xspan = (-1.0 , 1.0)
tspan = (0.0 , 1.0)
g(u , p , t) = 1
f(u , p , t) = 0
d = 1
sdealg = EM()
prob = SDEProblem(f , g , u0 , (0.0 , 1.0) ; xspan = xspan , d = d)
opt = Flux.ADAM(0.01)
m = Chain(Dense(1, 512, elu) , Dense(512 , 1024 , elu),Dense(1024 , 512 , elu), Dense(512 , 1))
sol = solve(prob, NeuralNetDiffEq.NNKolmogorov(m,opt , sdealg), verbose = true, dt = 0.001,
            abstol=1e-6, maxiters = 6)
T = tspan[2]
function analytical(xi)
    y = Float64[]
    div = 1/sqrt(2*pi)
    for x in xi
        y = push!(y , div*(exp(-(x^2)/2)))
    end
    y = reshape(y , size(xi)[1] , size(xi)[2] )
    return y
end
xs = xspan[1]:0.001:xspan[2]
x_val = rand(xs , d , 50)
errorl2 = Flux.mse(analytical(x_val) , m(x_val))
println("error_l2 = ", errorl2, "\n")
@test errorl2 < 0.3
