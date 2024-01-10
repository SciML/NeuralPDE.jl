using Flux, NeuralPDE, Test
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
import Lux

Random.seed!(110)

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
chain = Lux.Chain(Lux.Dense(1, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, 1))

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(initial_stepnorm = 0.01),
                         maxiters = 500)

func_approx(x) = discretization.phi(x, res.u)
@test func_approx(xs')≈func(xs') rtol=0.01

# plot(xs,func(xs))
# plot!(xs,func_approx(xs')')

## Approximation of function 1D 2
println("Approximation of function 1D 2")

@parameters x
@variables u(..)
func(x) = @. cos(5pi * x) * x
eq = [u(x) - u(0) ~ func(x)]
bc = [u(0) ~ u(0)]

x0 = 0
x_end = 4
domain = [x ∈ Interval(x0, x_end)]

hidden = 20
chain = Lux.Chain(Lux.Dense(1, hidden, Lux.sin),
                  Lux.Dense(hidden, hidden, Lux.sin),
                  Lux.Dense(hidden, hidden, Lux.sin),
                  Lux.Dense(hidden, 1))

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
prob = NeuralPDE.discretize(pde_system, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)

dx = 0.01
xs = collect(x0:dx:x_end)
func_s = func(xs)
func_approx(x) = discretization.phi(x, res.u) .- discretization.phi(0.0, res.u)
@test func_approx(xs')≈func(xs') rtol=0.01

# plot(xs,func(xs))
# plot!(xs, func_approx(xs')')

## Approximation of implicit function 1D
println("Approximation of implicit function 1D")

@parameters x
@variables u(..)
eq = [u(sin(x)) ~ cos(x) + cos(2 * x)]
bc = [u(0) ~ u(0)]

x0 = pi / 2
x_end = 3 * pi / 2
domain = [x ∈ Interval(x0, x_end)]

hidden = 20
chain = Lux.Chain(Lux.Dense(1, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, 1))

strategy = NeuralPDE.GridTraining(0.01)

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
prob = NeuralPDE.discretize(pde_system, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)

dy = 0.01
y0 = -1
y_end = 1
ys = collect(y0:dy:y_end)

func(y) = @. -sqrt(1 - y^2) + 1 - 2 * y^2
func_s = func(ys)
func_approx(y) = discretization.phi(y, res.u)
@test func_approx(ys')≈func(ys') rtol=0.01

# plot(ys,func(ys))
# plot!(ys, func_approx(ys')')

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
chain = Lux.Chain(Lux.Dense(2, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, 1))

strategy = NeuralPDE.GridTraining(d)
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
@named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
prob = NeuralPDE.discretize(pde_system, discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)
symprob.loss_functions.full_loss_function(symprob.flat_init_params, nothing)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 500)
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

## Approximation of function 2D
println("Approximation of function 2D 2")

@parameters x, y
@variables u(..)
func(x, y) = -sin(x) * sin(y) * exp(-((x - pi)^2 + (y - pi)^2))
eq = [u(x, y) - u(0, 0) ~ func(x, y)]
bc = [u(0, 0) ~ u(0, 0)]

x0 = -10
x_end = 10
y0 = -10
y_end = 10
d = 0.4

domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]

hidden = 25
chain = Lux.Chain(Lux.Dense(2, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, 1))

strategy = NeuralPDE.GridTraining(d)
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
@named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
prob = NeuralPDE.discretize(pde_system, discretization)
symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)
symprob.loss_functions.full_loss_function(symprob.flat_init_params, nothing)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)
phi = discretization.phi

xs = collect(x0:0.1:x_end)
ys = collect(y0:0.1:y_end)

function func_approx(x, y)
    first(phi([x, y], res.minimizer)) .- first(phi([0.0, 0.0], res.minimizer))
end
u_predict = reshape([func_approx(x, y) for x in xs for y in ys], (length(xs), length(ys)))
u_real = reshape([func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

@test u_predict≈u_real rtol=0.05
