import Pkg
Pkg.activate(temp=true)
Pkg.develop(path=".")
Pkg.add(["ModelingToolkit", "Lux", "DomainSets", "Optimization", "OptimizationOptimisers", "ComponentArrays", "Plots", "ModelingToolkitNeuralNets"])

using NeuralPDE
using ModelingToolkit
using DomainSets
using ComponentArrays
using Optimization
using OptimizationOptimisers
using Lux
using Plots
using Random

Random.seed!(42)

println("==== Symbolic PINN Parser Demo (MVP) ====")

# Advection Equation MVP problem:
# ∂u/∂t + 1.0 * ∂u/∂x = 0
# u(0, x) = sin(2pi * x)
# True solution: u(t, x) = sin(2pi * (x - t))

@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)

eq = Dt(u(t, x)) + 1.0 * Dx(u(t, x)) ~ 0

bcs = [
    u(0.0, x) ~ sin(2pi * x),
    u(t, 0.0) ~ sin(-2pi * t),
    u(t, 1.0) ~ sin(2pi * (1.0 - t))
]

domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)
]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Use the newly ported MVP parser 
loss_func, p_init, chain, st = build_pinn_loss(
    pde_system;
    width=6,
    depth=1,
    activation=tanh,
    n_points=10, 
    bc_weight=15.0,
    symbolic_expression_path="demo_symbolic_expression.txt"
)

# Convert to Optimization problem
obj = OptimizationFunction((theta, _) -> loss_func(theta), Optimization.AutoForwardDiff())
prob = OptimizationProblem(obj, collect(p_init), nothing)

println("Starting training for Advection Equation via Symbolic Loss Function...")
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters=1500)
println("Training complete. Final objective = ", res.objective)

# Evaluate and print comparison 
theta_final = ComponentArray(res.u, getaxes(p_init))

xs = collect(range(0.0, 1.0, length=10))
u_pred = [first(Lux.apply(chain, [0.5, xi], theta_final, st)[1]) for xi in xs]
u_true = [sin(2pi * (xi - 0.5)) for xi in xs]

println("\n=== Results ===")
println("Predictions: ", round.(u_pred, digits=4))
println("Exact:       ", round.(u_true, digits=4))
println("Max error:   ", round(maximum(abs.(u_pred .- u_true)), digits=4))
println("\nSymbolic PINN parser executed successfully!\n")
