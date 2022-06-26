using Flux, DiffEqFlux, NeuralPDE, Test
using Optimization, OptimizationOptimJL, OptimizationFlux
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random

Random.seed!(100)

## Approximation of function 1D
println("Approximation of function 1D")

@parameters x
@variables u(..)

func(x) = @. 2 + abs(x - 0.5)

eq = [u(x) ~ func(x)]
bc = [u(0) ~ u(0)]

x0 = 0
x_end = 2
dx = 0.001
domain = [x ∈ Interval(x0, x_end)]

xs = collect(x0:dx:x_end)
func_s = func(xs)

hidden = 10
chain = FastChain(FastDense(1, hidden, Flux.tanh),
                  FastDense(hidden, hidden, Flux.tanh),
                  FastDense(hidden, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy; initial_params = initθ)
@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

res = Optimization.solve(prob, ADAM(0.05), maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, BFGS(initial_stepnorm = 0.01), maxiters = 500)

@test discretization.phi(xs', res.u)≈func(xs') rtol=0.01

# plot(xs,func(xs))
# plot!(xs, discretization.phi(xs',res.u)')

## Approximation of function 1D 2
println("Approximation of function 1D 2")

@parameters x
@variables u(..)
func(x) = @. cos(5pi * x) * x
eq = [u(x) ~ func(x)]
bc = [u(0) ~ u(0)]

x0 = 0
x_end = 4
domain = [x ∈ Interval(x0, x_end)]

hidden = 20
chain = FastChain(FastDense(1, hidden, Flux.sin),
                  FastDense(hidden, hidden, Flux.sin),
                  FastDense(hidden, hidden, Flux.sin),
                  FastDense(hidden, 1))
initθ = DiffEqFlux.initial_params(chain)

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy; initial_params = initθ)
@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
prob = NeuralPDE.discretize(pde_system, discretization)

res = Optimization.solve(prob, ADAM(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, BFGS(), maxiters = 1000)

dx = 0.01
xs = collect(x0:dx:x_end)
func_s = func(xs)

@test discretization.phi(xs', res.u)≈func(xs') rtol=0.01

# plot(xs,func(xs))
# plot!(xs, discretization.phi(xs',res.u)')

## Approximation of function 2D
println("Approximation of function 2D")

@parameters x, y
@variables u(..)
func(x, y) = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))
eq = [u(x, y) ~ func(x, y)]
bc = [u(0, 0) ~ u(0, 0)]

x0 = -10
x_end = 10
y0 = -10
y_end = 10
d = 0.4

domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]

hidden = 25
chain = FastChain(FastDense(2, hidden, Flux.tanh),
                  FastDense(hidden, hidden, Flux.tanh),
                  FastDense(hidden, hidden, Flux.tanh),
                  FastDense(hidden, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))

strategy = NeuralPDE.GridTraining(d)
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy; initial_params = initθ)
@named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
prob = NeuralPDE.discretize(pde_system, discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)
discretized_functions = NeuralPDE.discretize_inner_functions(pde_system, discretization)
discretized_functions.full_loss_function(initθ, Float64[])

res = Optimization.solve(prob, ADAM(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, BFGS(), maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, BFGS(), maxiters = 500)
phi = discretization.phi

xs = collect(x0:0.1:x_end)
ys = collect(y0:0.1:y_end)
u_predict = reshape([first(phi([x, y], res.minimizer)) for x in xs for y in ys],
                    (length(xs), length(ys)))
u_real = reshape([func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict≈u_real rtol=0.05

# p1 = plot(xs, ys, u_real, st=:surface,title = "analytic");
# p2 = plot(xs, ys, u_predict, st=:surface,title = "predict");
# p3 = plot(xs, ys, diff_u,st=:surface,title = "error");
# plot(p1,p2,p3)
