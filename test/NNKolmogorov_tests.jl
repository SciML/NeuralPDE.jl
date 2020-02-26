using Test, Flux, NeuralNetDiffEq
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
g(u , p , t) = 0.5
f(u , p , t) = 0.5*0.25*u
d = 1
prob = KolmogorovPDEProblem(f , g, phi , tspan , xspan, d)
opt = Flux.Descent(0.1)
chain = Flux.Chain(Dense(1,128,tanh),Dense(128,256 ,tanh ) , Dense(256,128 ,tanh) ,
         Dense(128 , 1) )
sol = solve(prob, NeuralNetDiffEq.NNKolmogorov(chain,opt), verbose = true, dt = 0.1,
            abstol=1e-6, maxiters = 12)

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
real_sol = analytical(sol[1])
err = 0
for (r , s) in zip(real_sol , sol[2] )
    err  = ((r - s)^2 / s^2) + err
end
errorl2 = err/size(sol[2])[2]
println("error_l2 = ", errorl2, "\n")
@test errorl2 < 0.5
