using Test, Flux, Optim, DiffEqFlux, Optimization
using Random, NeuralPDE, DifferentialEquations
using Statistics, Distributions
import ModelingToolkit: Interval 
import DomainSets: UnitInterval
Random.seed!(100)

@parameters t z1 z2
@variables u(..)
Dt = Differential(t)

α = 1.2
β = 1.1

eq  = Dt(u(t,z1,z2))  ~ α*u(t,z1,z2) + β*u(t,z1,z2)*(√2*(z1*cos((1-1/2)*π*t) + z2*cos((2 - 1/2)*π*t)))
bcs = [u(0, z1, z2) ~ 1.0]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           z1 ∈ Interval(0.0,1.0), 
           z2 ∈ Interval(0.0,1.0)]

# number of dimensions
dim = 3
chain = Flux.Chain(Dense(dim,16,Flux.σ),Dense(16,16,Flux.σ),Dense(16,1))
# Initial parameters of Neural network
initθ = Float64.(DiffEqFlux.initial_params(chain))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain,GridTraining(dx),init_params =initθ)

@named pde_system = PDESystem(eq,bcs,domains,[t,z1,z2],[u(t,z1, z2)])
prob = discretize(pde_system,discretization)

opt = OptimizationOptimJL.BFGS()

#Callback function
callback = function (p,l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters=1000)
phi = discretization.phi

# Define analytic solution 
analytic_sol(u0,t,W) = u0*exp((α - β^2/2)*t + β*W)
u0 = 1.0

# Define truncated solition 
W_kkl(t, z1, z2) = √2*(z1*sin((1 - 1/2)*π*t)/((1-1/2)*π) + z2*sin((2 - 1/2)*π*t)/((2-1/2)*π))
truncated_sol(u0, t, z1, z2) = u0*exp((α - β^2/2)*t + β*W_kkl(t,z1,z2))

# Take samples of analytic solution and PINN solution
num_samples = 100
num_time_steps = dx/10
z1_samples = rand(Normal(), num_samples)
z2_samples = rand(Normal(), num_samples)
dt = dx/10
ts = 0:dt:1
num_time_steps = size(ts)[1]

W_samples = Array{Float64}(undef, num_time_steps, num_samples)
for i = 1:num_samples 
    W = WienerProcess(0.0, 0.0)
    prob = NoiseProblem(W,(0.0,1.0))
    sol = solve(prob;dt=dt)
    W_samples[:,i] = sol
end
analytic_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
predicted_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
truncated_solution_samples = Array{Float64}(undef, num_time_steps, num_samples)
for j = 1:num_samples
    for i = 1:num_time_steps 
        analytic_solution_samples[i,j] = analytic_sol(u0, ts[i], W_samples[i,j])
        predicted_solution_samples[i,j] = first(phi([ts[i],z1_samples[j], z2_samples[j]], res.minimizer))
        truncated_solution_samples[i,j] = truncated_sol(u0, ts[i], z1_samples[j], z2_samples[j])
    end
end

mean_analytic_solution = mean(analytic_solution_samples, dims = 2) 
mean_predicted_solution = mean(predicted_solution_samples, dims = 2)
mean_truncated_solution = mean(truncated_solution_samples, dims = 2)

using Plots
using Printf

p1 = plot(ts, mean_analytic_solution, title = @sprintf("Analytic Solution"))
p2 = plot(ts, mean_predicted_solution, title = @sprintf("PINN Predicted Solution"))
p3 = plot(ts, mean_truncated_solution, title = @sprintf("Truncated Solution"))
my_plot = plot(p1,p2,p3)
