using NeuralNetDiffEq
using Test

using Plots; plotly()
using DiffEqProblemLibrary, DiffEqDevTools, OrdinaryDiffEq
# include("NeuralNetDiffEq.jl")


#Example 1
linear = (u,p,t) -> t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3)))
linear_analytic = (u0,p,t) -> exp(-(t^2)/2)/(1+t+t^3) + t^2
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),1/2,(0.0,1.0))
dts = 1 ./ 2 .^ (10:-1:7)
sim = test_convergence(dts, prob, NeuralNetDiffEq.nnode())
@test abs(sim.𝒪est[:l2]) < 0.02
@test minimum(sim.errors[:l2]) < 0.4

# Example 2
linear = (u,p,t) -> -u/5 + exp(-t/5).*cos(t)
linear_analytic = (u0,p,t) ->  exp(-t/5)*(u0 + sin(t))
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),0.0,(0.0,1.0))
sim = test_convergence(dts, prob, NeuralNetDiffEq.nnode())
@test abs(sim.𝒪est[:l2]) < 0.7
@test minimum(sim.errors[:l2]) < 0.01

#Example 3
