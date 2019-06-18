using Test, Flux
using DiffEqDevTools

# Run a solve
linear = (u,p,t) -> cos(2pi*t)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0 ,tspan)
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt =Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.nnode(chain,opt), dt=1/20, abstol=1e-10, maxiters=300)
# println(sol)
# plot(sol)
# plot!(sol.t, t -> sin(2pi*t) / (2*pi), lw=3,ls=:dash,label="True Solution!")

#Example 1
# linear = (u,p,t) -> t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3)))
# linear_analytic = (u0,p,t) -> exp(-(t^2)/2)/(1+t+t^3) + t^2
# prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),1/2,(0.0,1.0))
# dts = 1 ./ 2 .^ (10:-1:7)
# sim = test_convergence(dts, prob, NeuralNetDiffEq.nnode(chain, opt))
# @test abs(sim.𝒪est[:l2]) < 0.1
# @test minimum(sim.errors[:l2]) < 0.5
#
# #Example 2
# linear = (u,p,t) -> -u/5 + exp(-t/5).*cos(t)
# linear_analytic = (u0,p,t) ->  exp(-t/5)*(u0 + sin(t))
# prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),0.0,(0.0,1.0))
# sim = test_convergence(dts, prob, NeuralNetDiffEq.nnode(chain, opt))
# @test abs(sim.𝒪est[:l2]) < 0.5
# @test minimum(sim.errors[:l2]) < 0.1
