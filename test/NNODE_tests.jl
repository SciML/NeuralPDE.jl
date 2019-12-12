ENV["GKS_ENCODING"]="utf-8"

using Test, Flux, NeuralNetDiffEq, Plots, Plots.PlotMeasures
using DiffEqDevTools


# Run a solve on vectors

linear = (u,p,t) -> [cos(2pi*t)]
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0 ,tspan)
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE(chain,opt), initial_condition = [u0], dt=1/20f0, abstol=1e-10,
           verbose = true, maxiters=200)


# Second-Order ODE - Dirichlet
linear = (u,p,t) -> cos(2pi*t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
u1 = 1.0f0
initial_condition = [u0, u1]
prob = ODEProblem(linear, u0, tspan)

chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE(chain,opt), initial_condition = [u0, u1], dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 200)

analytical_sol = t -> ((4pi^2 * t) - cos(2pi*t) + 1) / (4 * pi^2)
plot(analytical_sol, label = "analytical", linestyle = :dot, linewidth = 8, legend=:bottomright, bottom_margin = 10mm, left_margin=10mm, top_margin=10mm, right_margin=10mm)
plot!(sol, title = "Second-Order Dirichlet : Cos(2𝝅t)", label = "phi", fillalpha = .5, linewidth = 4)
xlabel!("Time (t)")
display(ylabel!("f(x)", ylims = (0,1)))
savefig("plot.png")

#=
anat_sum = zeros(0)
append!(anat_sum, 0)
for i = 1:20
    k = i * .05
    append!(anat_sum, analytical_sol(k))
end

print("length of sol is: ", length(sol))
print("length of anat_sum is:", length(anat_sum))
#error = sol .- analytical_sol
#print("Error: ", error)
#@test error < 0.5

=#

# Run a solve on vectors
linear = (u,p,t) -> [cos(2pi*t)]
tspan = (0.0f0, 1.0f0)
u0 = [0.0f0]
prob = ODEProblem(linear, u0 ,tspan)
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE(chain,opt), initial_condition = [u0], dt=1/20f0, abstol=1e-10,
            verbose = true, maxiters=200)

#Example 1
linear = (u,p,t) -> @. t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3)))
linear_analytic = (u0,p,t) -> [exp(-(t^2)/2)/(1+t+t^3) + t^2]
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),[1f0],(0.0f0,1.0f0))
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol  = solve(prob,NeuralNetDiffEq.NNODE(chain,opt),initial_condition = [prob.u0], verbose = true, dt=1/5f0)
err = sol.errors[:l2]
sol  = solve(prob,NeuralNetDiffEq.NNODE(chain,opt),initial_condition = [prob.u0], verbose = true, dt=1/20f0)
sol.errors[:l2]/err < 0.5

#=
dts = 1f0 ./ 2f0 .^ (6:-1:2)
sim = test_convergence(dts, prob, NeuralNetDiffEq.NNODE(chain, opt))
@test abs(sim.𝒪est[:l2]) < 0.1
@test minimum(sim.errors[:l2]) < 0.5
=#

#Example 2
linear = (u,p,t) -> -u/5 + exp(-t/5).*cos(t)
linear_analytic = (u0,p,t) ->  exp(-t/5)*(u0 + sin(t))
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),0.0f0,(0.0f0,1.0f0))
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol  = solve(prob,NeuralNetDiffEq.NNODE(chain,opt), initial_condition = [prob.u0], verbose = true, dt=1/5f0)
err = sol.errors[:l2]
sol  = solve(prob,NeuralNetDiffEq.NNODE(chain,opt),initial_condition = [prob.u0], verbose = true, dt=1/20f0)
sol.errors[:l2]/err < 0.5

#=
dts = 1f0 ./ 2f0 .^ (6:-1:2)
sim = test_convergence(dts, prob, NeuralNetDiffEq.NNODE(chain, opt))
@test abs(sim.𝒪est[:l2]) < 0.5
@test minimum(sim.errors[:l2]) < 0.1
=#
