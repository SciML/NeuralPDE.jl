using Flux
using DifferentialEquations
using LinearAlgebra
using Plots
using Statistics

NNODE = Chain(x -> [x],
           Dense(1, 32, tanh),
           Dense(32, 1),
           first)
NNODE(1.0)
g(t) = t * NNODE(t) + 1f0

ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t + ϵ) - g(t)) / ϵ) - cos(2π * t)) for t in 0:1f-2:1f0)
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () # callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss())
    end
end
display(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)

t = 0:0.001:1.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π .* t) / 2π, label="True")