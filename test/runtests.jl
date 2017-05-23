using NeuralNetDiffEq
using Base.Test

using Plots; plotly()
using DiffEqBase, ParameterizedFunctions
using DiffEqProblemLibrary, DiffEqDevTools


#Example 1
f = (t,u) -> t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3)))
@test eltype(t) == Float32 && eltype(u) == Float32
(::typeof(f))(::Type{Val{:analytic}},t,u0) =  exp(-(x^2)/2)/(1+x+x^3) + x^2
# prob = ODEProblem(f,Float32(0.0),(Float32(0.0),Float32(1.0)))
# sol = solve(prob,nnode(),dt=0.02)
dts = 1./2.^(14:-1:7)
prob = ODEProblem(f,0.0,(0.0,1.0))
sim  = test_convergence(dts,prob,odetf(),maxiters=Int(1e5))
@test abs(sim.𝒪est[:l2]-1) < 0.2
@test minimum(sim.errors[:l2]) < 0.002

#Example 2
f = (t,u) -> -u/5 + exp(-t/5).*cos(t)
@test eltype(t) == Float32 && eltype(u) == Float32
(::typeof(f))(::Type{Val{:analytic}},t,u0) =  exp(-t/5)*(u0 + sin(t))
prob = ODEProblem(f,Float32(0.0),(Float32(0.0),Float32(2.0)))
#sol = solve(prob,odetf(),dt=0.02)
#plot(sol,plot_analytic=true)
sim  = test_convergence(dts,prob,odetf(),maxiters=Int(1e5))
@test abs(sim.𝒪est[:l2]-1) < 0.2
@test minimum(sim.errors[:l2]) < 0.002

#Example 3
