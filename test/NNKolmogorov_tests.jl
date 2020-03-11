using Test, Flux, NeuralNetDiffEq , StochasticDiffEq
using DiffEqDevTools
#Initial case for u(0 , x)
function phi(xi)
    y = Float64[]
    for x in xi
        y = push!(y , 1.77*x -0.015*x^3)
    end
    y = reshape(y , size(xi)[1] , size(xi)[2] )
    return y
end

xspan = (-6.0 , 6.0)
tspan = (0.0 , 1.0)
#f = mu and g  = sigma
g(u , p , t) = 0.5*u
f(u , p , t) = 0.5*0.25*u
d = 1
sdealg = EM()
prob = KolmogorovPDEProblem(f , g, phi , xspan , tspan, d)
opt = Flux.ADAM(0.01)
m = Chain(Dense(1, 512, elu) , Dense(512 , 1024 , elu),Dense(1024 , 512 , elu), Dense(512 , 1))
sol = solve(prob, NeuralNetDiffEq.NNKolmogorov(m,opt , sdealg), verbose = true, dt = 0.001,
            abstol=1e-6, maxiters = 250)


function analytical(xi)
    y = Float64[]
    a = 1.77*exp(0.5*(0.5)^2*1.0)
    b = -0.015*exp(0.5*(0.5*3)^2*1.0)
    for x in xi
        y = push!(y , a*x + b*x^3)
    end
    y = reshape(y , size(xi)[1] , size(xi)[2] )
    return y
end
xs = xspan[1]:0.001:xspan[2]
x_val = rand(xs , d , 50)
errorl2 = Flux.mse(analytical(x_val) , m(x_val))
println("error_l2 = ", errorl2, "\n")
@test errorl2 < 0.3
