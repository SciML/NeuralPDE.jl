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
strategy_ = StochasticTraining(128, 128)
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

initθ = Float64.(DiffEqFlux.initial_params(chain_))
discretization = NeuralPDE.PhysicsInformedNN(chain_,
                                                strategy_;
                                                init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, ADAM(1e-3); maxiters=100)
phi = discretization.phi

xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real atol = 2.0

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)

inp = 1
eigs = 4
ts = range(0, 1, 1000)
begin
initW, initb = NeuralPDE.RectangularLaplacianEigenfunctionrMFNLayer(inp, eigs, 1, false, false, [Interval(0,1)], [(:neumann, :neumann)])
num_lap = inp * eigs
LEFastDense = DiffEqFlux.FastDense(inp, num_lap, sin; bias=true, initW=initW, initb=initb)
θ_0 = initial_params(LEFastDense)
apply_LE(t) = LEFastDense(t, θ_0)
vecmat = apply_LE.(ts)
transpose_vecmat = [[vecmat[t_j][i] for t_j in 1:length(ts)] for i in 1:eigs]
transpose_vecmat_sq = [(transpose_vecmat[i]).^3 for i in 1:eigs]
end
plot(ts, transpose_vecmat_sq)

inp = 2
eigs = 2
numts = 10
ts = range(0, 1, numts)
double_ts = map(x->[xs for xs in x], collect(IterTools.product(ts, ts)))
double_ts_flat = vec(double_ts)
input_matrix = [double_ts_flat[j][i] for i in 1:inp, j in 1:length(double_ts_flat)]
initW, initb = NeuralPDE.RectangularLaplacianEigenfunctionrMFNLayer(inp, eigs, 1, false, false, [Interval(0,1), Interval(0,1)], [(:dirichlet, :dirichlet), (:dirichlet, :dirichlet)])
num_lap = inp * eigs
LEFastDense = DiffEqFlux.FastDense(inp, num_lap, sin; bias=true, initW=initW, initb=initb)
θ_0 = initial_params(LEFastDense)
apply_LE(t) = LEFastDense(t, θ_0)
vecmat = apply_LE(input_matrix)
transpose_vecmat = [reshape(vecmat[i, :], numts, numts)  for i in 1:size(vecmat, 1)]
transpose_vecmat_sq = [(transpose_vecmat[i]).^3 for i in 1:eigs]
plot(ts, ts, transpose_vecmat[1]; linetype=:contourf)
nothing

