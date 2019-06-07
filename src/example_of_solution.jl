using DifferentialEquations, Plots

linear = (u,p,t) -> cos(2pi*t)
linear_analytic = (u0,p,t) -> u0*sin(2pi*t)/(2*pi)
tspan = (0.0,1.0)
u0 = 0.0
#ODEFunction(linear,analytic=linear_analytic)
prob = ODEProblem(linear, u0 ,tspan)
sol = NeuralNetDiffEq.solve(prob, NeuralNetDiffEq.nnode(5), dt=1/20, iterations=100)
# println(sol)
plot(sol)
plot!(sol.t, t -> sin(2pi*t) / (2*pi), lw=3,ls=:dash,label="True Solution!")
