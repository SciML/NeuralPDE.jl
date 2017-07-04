using NeuralNetDiffEq
using Base.Test

using Plots; plotly()
using DiffEqBase, ParameterizedFunctions
using DiffEqProblemLibrary, DiffEqDevTools

dts = 1./2.^(14:-1:7)
#Example 1
linear = (t,u) -> (1.01*u)
(f::typeof(linear))(::Type{Val{:analytic}},t,u0) = u0*exp(1.01*t)
prob = ODEProblem(linear,1/2,(0.0,1.0))
sol = solve(prob,nnode(10),dt=1/10,iterations=5000)

#Example 2
f = (t,u) -> (t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3))))
(::typeof(f))(::Type{Val{:analytic}},t,u0) =  u0*exp(-(t^2)/2)/(1+t+t^3) + t^2
prob2 = ODEProblem(f,1.0,(0.0,1.0))
sol2 = solve(prob2,nnode(10),dt=0.1,iterations=5000)
# prob = ODEProblem(f,Float32(0.0),(Float32(0.0),Float32(1.0)))
# sol = solve(prob,nnode(),dt=0.02)
prob = ODEProblem(f,0.0,(0.0,1.0))
sim  = test_convergence(dts,prob,odetf(),maxiters=Int(1e5))
@test abs(sim.𝒪est[:l2]-1) < 0.2
@test minimum(sim.errors[:l2]) < 0.002

#Example 3
f2 = (t,u) -> (-u/5 + exp(-t/5).*cos(t))
(::typeof(f2))(::Type{Val{:analytic}},t,u0) =  exp(-t/5)*(u0 + sin(t))
prob3 = ODEProblem(f2,Float32(0.0),(Float32(0.0),Float32(2.0)))
sol3 = solve(prob3,nnode(10),dt=0.2,iterations=1000)
#sol = solve(prob,odetf(),dt=0.02)
#plot(sol,plot_analytic=true)
sim  = test_convergence(dts,prob,odetf(),maxiters=Int(1e5))
@test abs(sim.𝒪est[:l2]-1) < 0.2
@test minimum(sim.errors[:l2]) < 0.002

#Example 4
