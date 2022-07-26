begin
using Flux
println("NNPDE_tests")
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Plots
using IterTools

using Random
Random.seed!(100)
println("Starting Soon!")

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
using DiffEqFlux, Flux, DiffEqBase, CUDA
end


chain_ = VectorOfrMFNChain([2], [32], [1], 4)[1]
strategy_ = StochasticTraining(128, 8)
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

# Initial and boundary conditions
bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
        u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
            y ∈ Interval(0.0,1.0)]

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])

initθ = Float64.(DiffEqFlux.initial_params(chain_))
discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                strategy_;
                                                init_params = initθ)

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, ADAM(1e-3); maxiters=10)
phi = discretization.phi

xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

#@test u_predict ≈ u_real atol = 2.0

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)


chain_domains = map(dom->dom.domain, domains) 
chain_boundary_conditions = [(:dirichlet, :dirichlet), (:dirichlet, :dirichlet)]
chains_LE_ = VectorOfrMFNChain([2], [5], [1], 5; laplacian_eigenfunctions=true,
    domains=chain_domains, boundary_conditions=chain_boundary_conditions)
chain_LE_ = chains_LE_[1]
initθ_LE = Float64.(DiffEqFlux.initial_params(chain_LE_))
"""
discretization_LE = NeuralPDE.PhysicsInformedNN(chain_LE_,
                                                strategy_;
                                                init_params = initθ_LE)
prob_LE = NeuralPDE.discretize(pde_system,discretization_LE)
sym_prob_LE = NeuralPDE.symbolic_discretize(pde_system,discretization_LE)
res_LE = GalacticOptim.solve(prob_LE, ADAM(3e-4); maxiters=10)
phi_LE = discretization_LE.phi
xis = Float32.([0 1 0.2; 1 0 0.4])
xi = Float32.([0, 1])
chain_LE_(xis, initθ_LE)
u_predict_LE_initial = reshape([first(phi_LE([x,y],initθ_LE)) for x in xs for y in ys],(length(xs),length(ys)))
u_predict_LE_final = reshape([first(phi_LE([x,y],res_LE)) for x in xs for y in ys],(length(xs),length(ys)))
diff_u_LE_initial = abs.(u_predict_LE_initial .- u_real)
diff_u_LE_final = abs.(u_predict_LE_final .- u_real)
p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict_LE_initial, linetype=:contourf,title = "predict_initial");
p3 = plot(xs, ys, u_predict_LE_final, linetype=:contourf,title = "predict_final");
p4 = plot(xs, ys, diff_u_LE_final,linetype=:contourf,title = "error");
plot(p1,p2,p3,p4)

"""

chains_mlp, init_params_mlp = NeuralPDE.VectorOfMLP([2], 16, 3, Flux.gelu, Flux.glorot_uniform)
chain_added = NeuralPDE.add_vector_fast_chains(chains_LE_, chains_mlp)[1]
initθ_added = Float64.(DiffEqFlux.initial_params(chain_added))
chain_added
chain_added(xis, initθ_added)
discretization_added = NeuralPDE.PhysicsInformedNN(chain_added,
                                                strategy_;
                                                init_params = initθ_added)
prob_added = NeuralPDE.discretize(pde_system,discretization_added)
sym_prob_added = NeuralPDE.symbolic_discretize(pde_system,discretization_added)
res_added = GalacticOptim.solve(prob_added, ADAM(3e-4); maxiters=100_000)
phi_added = discretization_added.phi
u_predict_added_initial = reshape([first(phi_added([x,y],initθ_added)) for x in xs for y in ys],(length(xs),length(ys)))
u_predict_added_final = reshape([first(phi_added([x,y],res_added)) for x in xs for y in ys],(length(xs),length(ys)))
diff_u_added_initial = abs.(u_predict_added_initial .- u_real)
diff_u_added_final = abs.(u_predict_added_final .- u_real)
p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict_added_initial, linetype=:contourf,title = "predict_initial");
p3 = plot(xs, ys, u_predict_added_final, linetype=:contourf,title = "predict_final");
p4 = plot(xs, ys, diff_u_added_final,linetype=:contourf,title = "error");
plot(p1,p2,p3,p4)

#nothing
