using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using CUDA

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
CUDA.allowscalar(false)
const gpuones = cu(ones(1))

## ODE
@parameters θ
@variables u(..)
Dθ = Differential(θ)

# 1D ODE
eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))

# Initial and boundary conditions
bcs = [u(0.) ~ 1.0]

# Space and time domains
domains = [θ ∈ IntervalDomain(0.0,1.0)]
# Discretization
dt = 0.1
# Neural network
inner = 12
chain = FastChain(FastDense(1,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,1),(u,p)->gpuones .* u) |> gpu
initθ = DiffEqFlux.initial_params(chain) |> gpu

strategy = NeuralPDE.GridTraining(dt)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = initθ
                                             )

pde_system = PDESystem(eq,bcs,domains,[θ],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system,discretization)
@time res = GalacticOptim.solve(prob, ADAM(1e-1); cb = cb, maxiters=1000)
phi = discretization.phi

analytic_sol_func(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
ts = [domain.domain.lower:dt/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(t) for t in ts]
u_predict  = [first(Array(phi(t,res.minimizer))) for t in ts]

@test u_predict ≈ u_real atol = 0.2

# t_plot = collect(ts)
# plot(t_plot ,u_real)
# plot!(t_plot ,u_predict)

## 1D PDE

@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
bcs = [u(0,x) ~ cos(x),
        u(t,0) ~ exp(-t),
        u(t,1) ~ exp(-t) * cos(1)]

domains = [t ∈ IntervalDomain(0.0,1.0),
          x ∈ IntervalDomain(0.0,1.0)]

pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

inner = 12
chain = FastChain(FastDense(2,inner,Flux.σ),
                  FastDense(12,12,Flux.σ),
                  FastDense(inner,1),(u,p)->gpuones .* u)

quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),
                                                   reltol = 1e-2, abstol = 1e-2,
                                                   maxiters = 50)
initθ = initial_params(chain) |>gpu
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             quadrature_strategy;
                                             init_params = initθ)
prob = NeuralPDE.discretize(pdesys,discretization)
symprob = NeuralPDE.symbolic_discretize(pdesys,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=1000)
phi = discretization.phi

ts,xs = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
u_predict = reshape([first(Array(phi([t,x],res.minimizer))) for t in ts for x in xs],(length(ts),length(xs)))
u_real = reshape([u_exact(t,x) for t in ts  for x in xs ], (length(ts),length(xs)))
diff_u = abs.(u_predict .- u_real)

@test u_predict ≈ u_real atol = 1.0

# p1 = plot(ts, xs, u_real, linetype=:contourf,title = "analytic");
# p2 = plot(ts, xs, u_predict, linetype=:contourf,title = "predict");
# p3 = plot(ts, xs, diff_u,linetype=:contourf,title = "error");
# plot(p1,p2,p3)
